"""
EPLB Modular Evaluator — subprocess-based for GIL-free parallelism.
Wraps the skydiscover EPLB evaluator.
"""
import sys
import os
import subprocess
import tempfile
import json
import time
import traceback

EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(EVALUATOR_DIR)
SUBPROCESS_TIMEOUT = int(os.environ.get("LEARNX_EPLB_EVAL_TIMEOUT", "360"))


def evaluate(program_path, temp_dir=None):
    """Evaluate EPLB program by spawning evaluator.py as subprocess."""
    from openevolve.evaluation_result import EvaluationResult
    from openevolve.modular_utils.error_constants import ErrorCodes
    from openevolve.modular_utils.score_transform import (
        create_score_transform_config_from_dict,
        transform_score_for_rl,
    )
    from openevolve.modular_utils.yaml_config_service import load_yaml_variables

    fd, output_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    # The skydiscover evaluator uses wrapper.py which reads sys.argv[1] as program path
    # and prints JSON to stdout
    eval_script = os.path.join(EVALUATOR_DIR, "evaluator.py")
    start_time = time.time()

    try:
        proc = subprocess.run(
            [sys.executable, eval_script, program_path],
            timeout=SUBPROCESS_TIMEOUT,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": EVALUATOR_DIR + ":" + os.environ.get("PYTHONPATH", "")},
            cwd=EVALUATOR_DIR,
        )
        eval_time = time.time() - start_time

        # The wrapper.py prints JSON to stdout
        stdout = proc.stdout.strip()
        try:
            data = json.loads(stdout)
        except Exception:
            # Try to find JSON in stdout (wrapper outputs JSON as last line)
            for line in reversed(stdout.split('\n')):
                line = line.strip()
                if line.startswith('{'):
                    try:
                        data = json.loads(line)
                        break
                    except Exception:
                        continue
            else:
                stderr_tail = (proc.stderr or "")[-2000:]
                return _error_result(
                    ErrorCodes.EXECUTION_ERROR, eval_time,
                    f"Failed to parse evaluation output. exit={proc.returncode} stderr: {stderr_tail}",
                )

        combined_score = data.get("combined_score", 0.0)
        error = data.get("error")

        if error and combined_score == 0.0:
            return _error_result(ErrorCodes.EXECUTION_ERROR, eval_time, error)

        # RL score transform
        score_transform_cfg = create_score_transform_config_from_dict(load_yaml_variables())
        rl_reward = transform_score_for_rl(combined_score, score_transform_cfg)

        metrics = {
            "combined_score": combined_score,
            "rl_normalized_reward": rl_reward,
            "eval_time": eval_time,
        }
        # Add any extra metrics from the evaluator
        for k, v in data.items():
            if k not in ("combined_score", "error") and isinstance(v, (int, float)):
                metrics[k] = v

        return EvaluationResult(metrics=metrics)

    except subprocess.TimeoutExpired:
        eval_time = time.time() - start_time
        return _error_result(ErrorCodes.TIMEOUT_ERROR, eval_time, f"Evaluation timed out after {SUBPROCESS_TIMEOUT}s")
    except Exception as e:
        eval_time = time.time() - start_time
        return _error_result(ErrorCodes.EXECUTION_ERROR, eval_time, f"Exception: {traceback.format_exc()[-1000:]}")
    finally:
        try:
            os.unlink(output_path)
        except Exception:
            pass


def _error_result(code, eval_time, message):
    from openevolve.evaluation_result import EvaluationResult
    return EvaluationResult(
        metrics={
            "combined_score": float(code),
            "validity": 0.0,
            "rl_normalized_reward": -0.5,
            "eval_time": eval_time,
            "error_code": code,
            "error": message[:500],
        }
    )
