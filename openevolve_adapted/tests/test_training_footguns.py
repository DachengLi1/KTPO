import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENEVOLVE_ROOT = REPO_ROOT / "openevolve_adapted"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingFootgunTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(OPENEVOLVE_ROOT))

    def tearDown(self):
        try:
            sys.path.remove(str(OPENEVOLVE_ROOT))
        except ValueError:
            pass

    def test_adrs_modular_errors_are_invalid_error_codes(self):
        from openevolve.modular_utils.error_constants import ErrorCodes

        evaluators = {
            "eplb": REPO_ROOT / "openevolve_adapted/examples/eplb/evaluators/evaluator_modular.py",
            "prism": REPO_ROOT / "openevolve_adapted/examples/prism/evaluators/evaluator_modular.py",
            "txn": REPO_ROOT / "openevolve_adapted/examples/txn_scheduling/evaluators/evaluator_modular.py",
        }

        for name, path in evaluators.items():
            with self.subTest(name=name):
                module = _load_module(path, f"{name}_modular_evaluator")
                result = module._error_result(ErrorCodes.EXECUTION_ERROR, 0.0, "boom")

                self.assertEqual(result.metrics["combined_score"], ErrorCodes.EXECUTION_ERROR)
                self.assertEqual(result.metrics["validity"], 0.0)
                self.assertEqual(result.metrics["rl_normalized_reward"], -0.5)

    def test_general_launchers_expose_acds_and_ray_footgun_knobs(self):
        for rel_path in [
            "scripts_launch/Qwen3-4B-Instruct-2507/_general.sh",
            "scripts_launch/Qwen3.5-9B/_general_colocate.sh",
        ]:
            with self.subTest(path=rel_path):
                text = (REPO_ROOT / rel_path).read_text()

                self.assertIn("RAY_memory_monitor_refresh_ms", text)
                self.assertIn("LEARNX_ACDS_VALID_ONLY_BEAT_RATE", text)
                self.assertIn("--evolving-gym-acds-valid-only-beat-rate", text)
                self.assertIn("LEARNX_ACDS_IMPROVING_ONLY", text)

    def test_eplb_launcher_checks_workload_before_training(self):
        text = (
            REPO_ROOT
            / "scripts_launch/Qwen3.5-9B/ktpo_qwen3.5_9B_eplb_nothink_colocate.sh"
        ).read_text()

        self.assertIn("expert-load.json", text)
        self.assertIn("huggingface.co/datasets/abmfy/eplb-openevolve", text)
        self.assertIn('if [ ! -f "${EPLB_WORKLOAD_FILE}" ]; then', text)


if __name__ == "__main__":
    unittest.main()
