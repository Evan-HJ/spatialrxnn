import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_experiment", SCRIPT)
run_experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_experiment)


class ExperimentPipelineTest(unittest.TestCase):
    def test_cli_arguments(self):
        self.assertEqual(
            run_experiment.cli_arguments(
                {"reaction": True, "quiet": False, "epochs": 2, "split_sizes": [0.8, 0.1, 0.1]}
            ),
            ["--reaction", "--epochs", "2", "--split_sizes", "0.8", "0.1", "0.1"],
        )

    def test_stage_dependencies_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "experiment.json"
            config_path.parent.mkdir()
            config = {
                "version": 1,
                "project_root": "..",
                "common": {"epochs": 100},
                "stages": [
                    {"name": "first", "data_path": "data/a.csv", "save_dir": "out/first"},
                    {
                        "name": "second",
                        "data_path": "data/b.csv",
                        "save_dir": "out/second",
                        "checkpoint_from": "first",
                    },
                ],
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            stages = run_experiment.prepare_stages(config_path, config, epochs=1)
            self.assertEqual(stages[0]["arguments"]["epochs"], 1)
            self.assertEqual(stages[0]["arguments"]["data_path"], str((root / "data/a.csv").resolve()))
            self.assertEqual(
                stages[1]["arguments"]["checkpoint_path"],
                str((root / "out/first/fold_0/model_0/model.pt").resolve()),
            )

    def test_rejects_later_dependency(self):
        config = {
            "version": 1,
            "stages": [
                {"name": "first", "data_path": "a.csv", "save_dir": "one", "checkpoint_from": "second"},
                {"name": "second", "data_path": "b.csv", "save_dir": "two"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "unknown or later"):
            run_experiment.prepare_stages(Path("config.json"), config)


if __name__ == "__main__":
    unittest.main()
