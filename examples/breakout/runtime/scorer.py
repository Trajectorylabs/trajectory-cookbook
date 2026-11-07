"""Fixed MinAtar Breakout: numeric weight edits, never contestant code execution."""
import ast
import re
import numpy as np
from minatar import Environment

SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32),
          "b2": (32,), "W3": (32, 6), "b3": (6,)}
MAX_STEPS = 10_000


def literal(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        if not isinstance(node.operand, ast.Constant):
            raise ValueError("Only signed numeric literals")
        return (-1 if isinstance(node.op, ast.USub) else 1) * literal(node.operand)
    if isinstance(node, ast.List):
        return [literal(x) for x in node.elts]
    raise ValueError("Only numeric literals and lists")


def index(node, size):
    def integer(n):
        value = literal(n)
        if type(value) is not int:
            raise ValueError("Index must be an integer")
        return value
    if isinstance(node, ast.Slice):
        start = 0 if node.lower is None else integer(node.lower)
        stop = size if node.upper is None else integer(node.upper)
        step = 1 if node.step is None else integer(node.step)
        if not 0 <= start < stop <= size or step <= 0:
            raise ValueError("Invalid slice")
        return slice(start, stop, step)
    value = integer(node)
    if not 0 <= value < size:
        raise ValueError("Index out of bounds")
    return value


def parse_weights(reply):
    if not isinstance(reply, str) or len(reply) > 300_000:
        raise ValueError("Reply exceeds parser budget")
    blocks = re.findall(r"```edits[ \t]*\n(.*?)```", reply, flags=re.S)
    if not blocks:
        raise ValueError("Missing complete edits block")
    weights = {name: np.zeros(shape, dtype=np.float32) for name, shape in SHAPES.items()}
    statements, pending, depth = [], [], 0
    for line in blocks[-1].splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        pending.append(line)
        depth += line.count("[") - line.count("]")
        if depth <= 0:
            statements.append(" ".join(pending))
            pending, depth = [], 0
    if pending:
        statements.append(" ".join(pending))
    if len(statements) > 4096:
        raise ValueError("Too many statements")
    applied, errors = 0, []
    for line_number, text in enumerate(statements, 1):
        try:
            body = ast.parse(text).body
            if len(body) != 1 or not isinstance(body[0], (ast.Assign, ast.AugAssign)):
                raise ValueError("Expected one numeric assignment")
            stmt = body[0]
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1:
                    raise ValueError("Expected one target")
                target = stmt.targets[0]
            else:
                target = stmt.target
            if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
                raise ValueError("Expected a tensor slice")
            name = target.value.id
            if name not in weights:
                raise ValueError("Unknown tensor")
            shape = SHAPES[name]
            axes = target.slice.elts if isinstance(target.slice, ast.Tuple) else [target.slice]
            if len(axes) != len(shape):
                raise ValueError("Wrong index dimensions")
            selection = tuple(index(n, size) for n, size in zip(axes, shape))
            before = weights[name][selection]
            value = np.asarray(literal(stmt.value), dtype=np.float64)
            if value.shape not in ((), before.shape) or not np.isfinite(value).all() or np.any(np.abs(value) > 1e6):
                raise ValueError("Invalid value shape or magnitude")
            if isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.op, ast.Add):
                    value = before.astype(np.float64) + value
                elif isinstance(stmt.op, ast.Sub):
                    value = before.astype(np.float64) - value
                elif isinstance(stmt.op, ast.Mult):
                    value = before.astype(np.float64) * value
                else:
                    raise ValueError("Unsupported operation")
            if not np.isfinite(value).all() or np.any(np.abs(value) > 1e6):
                raise ValueError("Result exceeds weight bounds")
            weights[name][selection] = value
            applied += 1
        except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError):
            errors.append(line_number)
    if not applied:
        raise ValueError("No readable numeric assignments")
    return weights, {"statements": len(statements), "applied": applied,
                     "skipped": errors, "nonzero_weights": sum(int(np.count_nonzero(w)) for w in weights.values())}


def observation(state):
    x = np.zeros((10, 10, 6), dtype=np.float32)
    x[:, :, [0, 2, 3, 4]] = state[:, :, [0, 3, 1, 2]]
    return x.reshape(600)


def logits(weights, x):
    h1 = np.tanh(x @ weights["W1"] + weights["b1"])
    h2 = np.tanh(h1 @ weights["W2"] + weights["b2"])
    return h2 @ weights["W3"] + weights["b3"]


def evaluate(weights, seeds, max_steps=MAX_STEPS):
    """Separate RNG per episode and for policy sampling; deterministic replay."""
    if not seeds or len(seeds) != len(set(seeds)) or not 1 <= max_steps <= MAX_STEPS:
        raise ValueError("Invalid evaluation seeds or step limit")
    games, rngs = [], []
    for seed in seeds:
        game = Environment("breakout", sticky_action_prob=0.1)
        game.seed(seed)
        game.reset()
        games.append(game)
        rngs.append(np.random.RandomState(seed ^ 0x5A17B3C9))
    scores = np.zeros(len(seeds), dtype=np.int64)
    lengths = np.zeros(len(seeds), dtype=np.int64)
    terminal = np.zeros(len(seeds), dtype=bool)
    for _ in range(max_steps):
        active = np.flatnonzero(~terminal)
        if not len(active):
            break
        x = np.stack([observation(games[i].state()) for i in active])
        z = logits(weights, x).astype(np.float64)
        z -= z.max(axis=1, keepdims=True)
        probs = np.exp(z)
        probs /= probs.sum(axis=1, keepdims=True)
        cdf = np.cumsum(probs, axis=1)
        cdf[:, -1] = 1.0
        for row, i in enumerate(active):
            action = int(np.searchsorted(cdf[row], rngs[i].rand(), side="right"))
            reward, done = games[i].act(action)
            scores[i] += reward
            lengths[i] += 1
            terminal[i] = done
    return {"episodes": len(seeds), "mean_return": float(scores.mean()),
            "standard_error": float(scores.std(ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else None,
            "mean_length": float(lengths.mean()), "cap_hits": int((~terminal).sum()),
            "returns": scores.tolist(), "lengths": lengths.tolist(), "seeds": seeds}
