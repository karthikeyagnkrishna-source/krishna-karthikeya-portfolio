import subprocess
import sys
import unittest


class ExportP2000EntrypointTest(unittest.TestCase):
    def test_help_runs_from_project_root(self):
        result = subprocess.run(
            [sys.executable, "scripts/export_shared_p2000_k1.py", "--help"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-frames", result.stdout)
        self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()
