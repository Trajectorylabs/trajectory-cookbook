"""Run model Python in a fresh, unprivileged Linux process with no filesystem or network access."""

import json
import os
import platform
import selectors
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import chess

MAX_CODE_BYTES = 16 * 1024
MAX_OUTPUT_BYTES = 16 * 1024
WALL_SECONDS = 15
CPU_SECONDS = 10
MEMORY_BYTES = 256 * 1024 * 1024

_WORKER = r"""
import collections, copy, ctypes, datetime, errno, functools, gc, heapq
import importlib.util, io, itertools, json, math, os, random, re, resource
import statistics, sys, time, traceback, typing

status_fd = int(sys.argv[3])
try:
    os.environ.clear()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (268435456, 268435456))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16384, 16384))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    payload = json.load(sys.stdin)
    spec = importlib.util.spec_from_file_location("chess", sys.argv[2], submodule_search_locations=[os.path.dirname(sys.argv[2])])
    chess = importlib.util.module_from_spec(spec)
    sys.modules["chess"] = chess
    spec.loader.exec_module(chess)
    board = chess.Board(payload["fen"])
    root_moves = payload["root_moves"]
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    maximum_fd = os.sysconf("SC_OPEN_MAX")
    os.closerange(3, status_fd)
    os.closerange(status_fd + 1, maximum_fd)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    os.chroot(sys.argv[1])
    os.chdir("/")
    os.setgroups([])
    os.setresgid(65534, 65534, 65534)
    os.setresuid(65534, 65534, 65534)
    if os.getresuid() != (65534, 65534, 65534) or os.getresgid() != (65534, 65534, 65534) or os.getgroups():
        raise RuntimeError("Privilege drop did not complete")
    class CapabilityHeader(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint), ("pid", ctypes.c_int)]

    class CapabilityData(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint), ("permitted", ctypes.c_uint), ("inheritable", ctypes.c_uint)]

    capability_header = CapabilityHeader(0x20080522, 0)
    capability_data = (CapabilityData * 2)()
    if libc.capset(ctypes.byref(capability_header), ctypes.byref(capability_data)) != 0:
        raise OSError(ctypes.get_errno(), "capability drop unavailable")
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "no_new_privs unavailable")

    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]

    class Program(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(Filter))]

    # x86_64 compute, signal, clock, memory and stdio operations only. Other ABIs are killed.
    allowed = (0, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 23, 24, 25, 28,
               35, 39, 60, 72, 96, 97, 99, 102, 104, 107, 108, 115, 118, 120, 127, 131, 186, 202, 219,
               228, 229, 230, 231, 270, 271, 273, 318, 334)
    instructions = [Filter(0x20, 0, 0, 4), Filter(0x15, 1, 0, 0xc000003e),
                    Filter(0x06, 0, 0, 0x80000000), Filter(0x20, 0, 0, 0)]
    for syscall in allowed:
        instructions.extend((Filter(0x15, 0, 1, syscall), Filter(0x06, 0, 0, 0x7fff0000)))
    instructions.append(Filter(0x06, 0, 0, 0x00050000 | errno.EPERM))
    filters = (Filter * len(instructions))(*instructions)
    program = Program(len(filters), filters)
    if libc.prctl(22, 2, ctypes.cast(ctypes.pointer(program), ctypes.c_void_p).value, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "seccomp unavailable")
    os.write(status_fd, b"isolated\n")
    os.close(status_fd)
except BaseException as error:
    print("Python isolation unavailable: " + type(error).__name__ + ": " + str(error), file=sys.stderr)
    sys.exit(70)

try:
    exec(compile(payload["code"], "<python_chess>", "exec"), {"__name__": "python_chess", "chess": chess, "board": board, "root_moves": root_moves})
except BaseException:
    traceback.print_exc()
    sys.exit(1)
"""


def execute_chess_python(fen: str, root_moves: dict[str, str], code: str) -> dict:
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty Python string")
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise ValueError(f"Python code exceeds {MAX_CODE_BYTES} UTF-8 bytes")
    result = {
        "stdout": "",
        "stderr": "",
        "exit_code": None,
        "isolation_active": False,
        "positions_searched": None,
        "limits": {
            "wall_seconds": WALL_SECONDS,
            "cpu_seconds": CPU_SECONDS,
            "memory_bytes": MEMORY_BYTES,
            "combined_output_bytes": MAX_OUTPUT_BYTES,
            "code_bytes": MAX_CODE_BYTES,
        },
    }
    if sys.platform != "linux" or platform.machine() != "x86_64" or os.geteuid() != 0:
        result["error"] = {
            "kind": "isolation_unavailable",
            "message": "Python isolation requires Linux x86_64 with chroot and privilege-drop support",
        }
        return result
    started = time.monotonic()
    status_read, status_write = os.pipe()
    process = None
    try:
        with (
            tempfile.TemporaryDirectory(prefix="chess-python-") as jail,
            tempfile.TemporaryFile() as source,
        ):
            Path(jail).chmod(0o555)
            source.write(
                json.dumps(
                    {"fen": fen, "root_moves": root_moves, "code": code}
                ).encode()
            )
            source.seek(0)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    _WORKER,
                    jail,
                    str(Path(chess.__file__).resolve()),
                    str(status_write),
                ],
                stdin=source,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={},
                close_fds=True,
                pass_fds=(status_write,),
                start_new_session=True,
            )
            os.close(status_write)
            status_write = None
            buffers = {
                "stdout": bytearray(),
                "stderr": bytearray(),
                "status": bytearray(),
            }
            output_bytes = 0
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                selector.register(status_read, selectors.EVENT_READ, "status")
                while selector.get_map():
                    remaining = WALL_SECONDS - (time.monotonic() - started)
                    if remaining <= 0:
                        result["error"] = {
                            "kind": "timeout",
                            "message": f"Python exceeded {WALL_SECONDS} wall seconds",
                        }
                        break
                    for key, _ in selector.select(remaining):
                        data = os.read(
                            key.fd,
                            128
                            if key.data == "status"
                            else min(4096, MAX_OUTPUT_BYTES - output_bytes + 1),
                        )
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        buffers[key.data].extend(data)
                        if key.data != "status":
                            output_bytes += len(data)
                        if output_bytes > MAX_OUTPUT_BYTES:
                            result["error"] = {
                                "kind": "output_limit",
                                "message": f"Combined Python stdout/stderr exceeded {MAX_OUTPUT_BYTES} bytes; output was discarded",
                            }
                            break
                    if "error" in result:
                        break
            if "error" in result:
                process.kill()
            try:
                process.wait(
                    timeout=max(0.01, WALL_SECONDS - (time.monotonic() - started))
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                result["error"] = {
                    "kind": "timeout",
                    "message": f"Python exceeded {WALL_SECONDS} wall seconds",
                }
            result["process_exit_code"] = process.returncode
            result["isolation_active"] = buffers["status"] == b"isolated\n"
            if result.get("error", {}).get("kind") != "output_limit":
                try:
                    for name in ("stdout", "stderr"):
                        result[name] = buffers[name].decode("utf-8")
                except UnicodeDecodeError:
                    result["stdout"] = result["stderr"] = ""
                    result["error"] = {
                        "kind": "invalid_output_encoding",
                        "message": "Python output must be UTF-8 text; invalid output was discarded",
                    }
            if not result["isolation_active"]:
                result["error"] = {
                    "kind": "isolation_unavailable",
                    "message": "Python isolation did not complete; model code was not run",
                }
            elif process.returncode != 0 and "error" not in result:
                result["error"] = {
                    "kind": "python_error",
                    "message": f"Python exited with status {process.returncode}",
                }
            result["exit_code"] = (
                process.returncode
                if process.returncode != 0 or "error" not in result
                else 1
            )
            return result
    except OSError as error:
        result["error"] = {
            "kind": "isolation_unavailable",
            "message": f"Python isolation could not start: {type(error).__name__}: {error}",
        }
        return result
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        os.close(status_read)
        if status_write is not None:
            os.close(status_write)
