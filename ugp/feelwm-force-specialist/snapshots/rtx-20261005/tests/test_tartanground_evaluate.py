import json
import tempfile
import unittest
from pathlib import Path

from tartanground.evaluate import STRICT_HOLDOUTS, validate_strict_split


class StrictTrajectorySplitTests(unittest.TestCase):
    def _manifest(self, directory: str, name: str, trajectories: set[str]) -> Path:
        path = Path(directory) / f"{name}.jsonl"
        path.write_text("".join(json.dumps({"trajectory_id": item}) + "\n" for item in sorted(trajectories)))
        return path

    def test_allows_temporal_train_dev_split_but_keeps_test_unseen(self):
        with tempfile.TemporaryDirectory() as directory:
            train = self._manifest(directory, "train", {"ForestEnv/P2000"})
            dev = self._manifest(directory, "dev", {"ForestEnv/P2000"})
            test = self._manifest(directory, "test", STRICT_HOLDOUTS)
            validate_strict_split(train, dev, test)

    def test_rejects_any_test_trajectory_seen_in_train_or_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            train = self._manifest(directory, "train", {"ForestEnv/P2000"})
            dev = self._manifest(directory, "dev", set())
            test = self._manifest(directory, "test", STRICT_HOLDOUTS | {"ForestEnv/P2000"})
            with self.assertRaisesRegex(ValueError, "strict test trajectories must be unseen"):
                validate_strict_split(train, dev, test)


if __name__ == "__main__":
    unittest.main()
