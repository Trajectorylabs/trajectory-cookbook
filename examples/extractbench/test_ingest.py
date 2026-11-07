"""Offline packaging checks using a small synthetic manifest, never benchmark answers."""

import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

module_spec = importlib.util.spec_from_file_location("extractbench_ingest", Path(__file__).with_name("ingest.py"))
ingest = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(ingest)


class IngestTests(unittest.TestCase):
    def fixture(self, root):
        name = "short--fixture"
        task = root / "source/tasks" / name
        (task / "environment").mkdir(parents=True)
        (task / "tests").mkdir()
        (task / "solution").mkdir()
        (task / "task.toml").write_text('''schema_version = "1.3"
artifacts = ["/app/output.json"]
[metadata]
source_case_id = "short/fixture"
source_split = "short"
[agent]
timeout_sec = 1800
network_mode = "allowlist"
allowed_hosts = []
[environment]
cpus = 2
memory_mb = 4096
[verifier]
environment_mode = "separate"
timeout_sec = 1800
[verifier.environment]
cpus = 2
memory_mb = 8192
network_mode = "no-network"
''')
        for name_, text in {
            "environment/Dockerfile": "COPY document.pdf schema.json /app/\n",
            "environment/document.pdf": "%PDF synthetic fixture",
            "environment/schema.json": "{}",
            "instruction.md": "Write output.json",
            "tests/reference.test.json": "SYNTHETIC_PRIVATE_REFERENCE",
            "tests/test.sh": "# fixture",
            "tests/Dockerfile": "COPY . /tests/",
            "solution/solve.sh": "SYNTHETIC_SOLUTION",
        }.items():
            (task / name_).write_text(text)
        record = {"case_id": "short/fixture", "files": {
            str(path.relative_to(task)): ingest.sha256(path)
            for path in task.rglob("*") if path.is_file()}}
        manifest = {"tasks": {name: record}}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return root / "source", name, record, manifest_path, manifest

    def test_source_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, name, record, _, _ = self.fixture(Path(temporary))
            (source / "tasks" / name / "instruction.md").write_text("changed")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                ingest.task_config(source, name, record)

    def test_archive_excludes_solution_and_keeps_private_tests_outside_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, name, _, manifest_path, manifest = self.fixture(root)
            with patch.object(ingest, "source_manifest", return_value=(manifest_path, manifest)):
                context, spec = ingest.stage(root / "stage", [name], False, source)
            with tarfile.open(context / "tasks.tar.gz") as archive:
                names = archive.getnames()
            self.assertFalse(any("solution/" in name for name in names))
            self.assertIn(f"tasks/{name}/tests/reference.test.json", names)
            self.assertFalse(any("reference" in name and "/environment/" in name for name in names))
            self.assertEqual(spec.tasks[0].split, "test")
            self.assertTrue(spec.tasks[0].spec["do_not_train"])
            self.assertEqual(spec.tasks[0].env_resources.memory_mb, 12288)

    def test_unpublished_task_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _, _, manifest_path, manifest = self.fixture(root)
            with patch.object(ingest, "source_manifest", return_value=(manifest_path, manifest)):
                with self.assertRaisesRegex(ValueError, "published task"):
                    ingest.stage(root / "stage", ["unpublished"], False, source)

    def test_default_selects_short_instead_of_lexicographically_first_long(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, name, record, manifest_path, manifest = self.fixture(root)
            manifest["tasks"]["long--first"] = record
            with patch.object(ingest, "source_manifest", return_value=(manifest_path, manifest)):
                _, spec = ingest.stage(root / "stage", [], False, source)
            self.assertEqual(spec.tasks[0].name, f"extractbench/{name}")

    def test_verifier_smoke_is_a_separate_explicit_spec(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, name, _, manifest_path, manifest = self.fixture(root)
            with patch.object(ingest, "source_manifest", return_value=(manifest_path, manifest)):
                _, spec = ingest.stage(root / "stage", [name], False, source, verifier_smoke=True)
            self.assertEqual(spec.name, "extractbench-harbor-verifier-check")
            self.assertTrue(spec.tasks[0].run_command.endswith(" --verifier-smoke"))
            self.assertTrue(spec.tasks[0].spec["verifier_smoke"])

    def test_relaxed_source_network_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, name, record, _, _ = self.fixture(Path(temporary))
            config_path = source / "tasks" / name / "task.toml"
            config_path.write_text(config_path.read_text().replace('network_mode = "allowlist"', 'network_mode = "public"'))
            record["files"]["task.toml"] = ingest.sha256(config_path)
            with self.assertRaisesRegex(ValueError, "isolation"):
                ingest.task_config(source, name, record)


if __name__ == "__main__":
    unittest.main()
