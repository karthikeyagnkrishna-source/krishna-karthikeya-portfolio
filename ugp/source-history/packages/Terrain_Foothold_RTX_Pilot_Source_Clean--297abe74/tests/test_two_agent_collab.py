import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import two_agent_collab as collab


class CollaborationPreflightTests(unittest.TestCase):
    def test_git_repo_without_commit_is_not_a_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            for name in collab.REQUIRED:
                (root / name).write_text("source", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            real_which = shutil.which
            with patch.object(collab.shutil, "which", side_effect=lambda binary: "codex" if binary == "codex" else real_which(binary)):
                _, problems = collab._check(root, "codex")
            self.assertTrue(any("baseline commit" in problem for problem in problems), problems)

    def test_claude_review_command_uses_supported_read_only_flags(self):
        cmd = collab._reviewer_command(pathlib.Path("/project"), pathlib.Path("/report"), "claude", None)
        self.assertNotIn("--restricted", cmd)
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "plan")
        self.assertEqual(cmd[cmd.index("--tools") + 1], "Read,Glob,Grep")

    def test_failed_smoke_returns_failure_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            report = root / "smoke.txt"
            with self.assertRaises(RuntimeError):
                collab._run_smoke([sys.executable, "-c", "import sys; sys.exit(7)"], root, 30, report)
            self.assertIn("Exit code 7", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
