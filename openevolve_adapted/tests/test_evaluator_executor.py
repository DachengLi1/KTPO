import asyncio
import os
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openevolve.config import EvaluatorConfig
from openevolve.evaluator import Evaluator


def _write_eval_file() -> str:
    fd, path = tempfile.mkstemp(suffix=".py", text=True)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            textwrap.dedent(
                """
                def evaluate(program_path, temp_dir=None):
                    return {"combined_score": 1.0}
                """
            )
        )
    return path


class EvaluatorExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.eval_file = _write_eval_file()

    def tearDown(self) -> None:
        if os.path.exists(self.eval_file):
            os.remove(self.eval_file)

    def test_constructor_accepts_explicit_executor_size(self) -> None:
        evaluator = Evaluator(
            EvaluatorConfig(parallel_evaluations=2, cascade_evaluation=False),
            self.eval_file,
            executor_max_workers=7,
        )
        self.assertEqual(evaluator._executor._max_workers, 7)
        evaluator._executor.shutdown(wait=True)

    def test_direct_evaluate_uses_explicit_executor(self) -> None:
        evaluator = Evaluator(
            EvaluatorConfig(parallel_evaluations=2, timeout=5, cascade_evaluation=False),
            self.eval_file,
            executor_max_workers=5,
        )

        captured = {}

        async def _run() -> None:
            fake_future = asyncio.get_running_loop().create_future()
            fake_future.set_result({"combined_score": 1.0})

            def _run_in_executor(executor, func, *args):
                captured["executor"] = executor
                return fake_future

            fake_loop = SimpleNamespace(run_in_executor=_run_in_executor)
            with patch("openevolve.evaluator.asyncio.get_event_loop", return_value=fake_loop):
                result = await evaluator._direct_evaluate("/tmp/program.py")
            self.assertEqual(result["combined_score"], 1.0)

        asyncio.run(_run())
        self.assertIs(captured["executor"], evaluator._executor)
        evaluator._executor.shutdown(wait=True)

    def test_collect_runtime_environment_reports_stats(self) -> None:
        evaluator = Evaluator(
            EvaluatorConfig(
                parallel_evaluations=2,
                cascade_evaluation=False,
                collect_runtime_environments=True,
                runtime_environment_patterns=["*"],
            ),
            self.eval_file,
            executor_max_workers=2,
        )

        program_id = "prog-1234"
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "output.txt"), "w", encoding="utf-8") as f:
                f.write("ok")
            os.makedirs(os.path.join(temp_dir, "results"), exist_ok=True)
            with open(os.path.join(temp_dir, "results", "child.txt"), "w", encoding="utf-8") as f:
                f.write("nested")

            with self.assertLogs("openevolve.evaluator", level="INFO") as logs:
                stats = evaluator._collect_runtime_environment(temp_dir, program_id)

            self.assertEqual(stats["program_id"], program_id)
            self.assertEqual(stats["collected_files"], 2)
            self.assertGreaterEqual(stats["elapsed_s"], 0.0)
            self.assertIn("[RuntimeEnv]", "\n".join(logs.output))

            runtime_env_dir = evaluator.get_pending_runtime_environment(program_id, clear=True)
            self.assertIsNotNone(runtime_env_dir)
            if runtime_env_dir and os.path.exists(runtime_env_dir):
                import shutil

                shutil.rmtree(runtime_env_dir, ignore_errors=True)

        evaluator._executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
