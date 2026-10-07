"""
LLM-SQL Modular Evaluator — subprocess-based for GIL-free parallelism.

Spawns verify.py as a subprocess so 192+ concurrent evaluations run truly
in parallel (each in its own Python process), unlike in-process evaluation
which is GIL-bound.
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

VERIFY_SCRIPT = os.path.join(EVALUATOR_DIR, "verify.py")
PER_DATASET_TIMEOUT = int(os.environ.get("LEARNX_LLM_SQL_DS_TIMEOUT", "60"))
SUBPROCESS_TIMEOUT = int(os.environ.get("LEARNX_LLM_SQL_EVAL_TIMEOUT", "120"))


def evaluate(program_path, temp_dir=None):
    """Evaluate LLM-SQL program by spawning verify.py as a subprocess."""
    from openevolve.evaluation_result import EvaluationResult
    from openevolve.modular_utils.error_constants import ErrorCodes
    from openevolve.modular_utils.score_transform import (
        create_score_transform_config_from_dict,
        transform_score_for_rl,
    )
    from openevolve.modular_utils.yaml_config_service import load_yaml_variables

    fd, output_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    start_time = time.time()

    try:
        proc = subprocess.run(
            [
                sys.executable, VERIFY_SCRIPT,
                "--program", program_path,
                "--output", output_path,
                "--per-dataset-timeout", str(PER_DATASET_TIMEOUT),
            ],
            timeout=SUBPROCESS_TIMEOUT,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        eval_time = time.time() - start_time

        # Parse JSON result from verify.py
        try:
            with open(output_path) as f:
                data = json.load(f)
        except Exception:
            stderr_tail = (proc.stderr or "")[-2000:]
            return _error_result(
                ErrorCodes.EXECUTION_ERROR, eval_time,
                f"Failed to parse evaluation output. exit={proc.returncode} stderr: {stderr_tail}",
            )

        # Complete failure (program load error, missing Solution, etc.)
        if "error" in data and data.get("combined_score", 0.0) == 0.0 and "per_dataset" not in data:
            return _error_result(
                ErrorCodes.EXECUTION_ERROR, eval_time, data["error"],
            )

        combined_score = data.get("combined_score", 0.0)
        avg_hit_rate = data.get("average_hit_rate", 0.0)
        avg_runtime = data.get("average_runtime", 0.0)
        num_ds = data.get("num_datasets", 5)
        num_ok = data.get("num_succeeded", 0)

        # RL score transform
        score_transform_cfg = create_score_transform_config_from_dict(load_yaml_variables())
        rl_reward = transform_score_for_rl(combined_score, score_transform_cfg)

        # Per-dataset metrics for logging
        extra = {}
        for ds_name, ds_data in data.get("per_dataset", {}).items():
            key = "".join(ch if ch.isalnum() else "_" for ch in ds_name).strip("_")
            if isinstance(ds_data, dict):
                if "hit_rate" in ds_data:
                    extra[f"hit_rate_{key}"] = ds_data["hit_rate"]
                if "runtime" in ds_data:
                    extra[f"runtime_{key}_s"] = ds_data["runtime"]

        metrics = {
            "combined_score": combined_score,
            "rl_normalized_reward": rl_reward,
            "validity": float(num_ok) / num_ds if num_ds > 0 else 0.0,
            "objective_value": avg_hit_rate,
            "average_runtime": avg_runtime,
            "total_runtime": data.get("total_runtime", 0.0),
            "max_runtime_s": max(data.get("runtimes", [0.0])),
            "eval_time": eval_time,
            **extra,
        }

        # Artifacts: per-dataset feedback for multistep refinement
        artifacts = {}
        feedback_lines = _format_per_dataset_feedback(data)
        if feedback_lines:
            artifacts["error"] = feedback_lines

        # Include subprocess stdout as program_output for debugging
        if proc.stdout:
            artifacts["program_output"] = proc.stdout[-4000:]

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    except subprocess.TimeoutExpired:
        eval_time = time.time() - start_time
        return _error_result(
            ErrorCodes.EXECUTION_ERROR, eval_time,
            f"Evaluation subprocess timed out after {SUBPROCESS_TIMEOUT}s",
        )
    except Exception as e:
        eval_time = time.time() - start_time
        return _error_result(
            ErrorCodes.EXECUTION_ERROR, eval_time,
            f"Evaluation failed: {e}\n{traceback.format_exc()[-1000:]}",
        )
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


DATASET_INFO = {
    "movies":   {"rows": 15000, "cols": 6,  "note": "1 col_merge group of 3 cols"},
    "beer":     {"rows": 28479, "cols": 8,  "note": "1 col_merge group of 2 cols"},
    "bird":     {"rows": 14920, "cols": 3,  "note": "1 col_merge group of 2 cols"},
    "pdmx":     {"rows": 10000, "cols": 51, "note": "2 col_merge groups (2 cols + 6 cols)"},
    "products": {"rows": 14890, "cols": 7,  "note": "1 col_merge group of 2 cols"},
}


def _format_per_dataset_feedback(data):
    """Build a concise per-dataset report for multistep feedback."""
    per_ds = data.get("per_dataset", {})
    if not per_ds:
        return None

    lines = ["Evaluated on 5 representative datasets (per-dataset timeout=60s). Do NOT hardcode dataset-specific logic; the final evaluation uses different datasets:"]
    for ds_name in ["movies", "beer", "bird", "pdmx", "products"]:
        ds_data = per_ds.get(ds_name, {})
        info = DATASET_INFO.get(ds_name, {})
        size_tag = f"{info.get('rows', '?')}rows×{info.get('cols', '?')}cols"
        if info.get("note"):
            size_tag += f", {info['note']}"

        if "error" in ds_data:
            err = ds_data["error"]
            if len(err) > 120:
                err = err[:120] + "..."
            lines.append(f"  {ds_name} ({size_tag}): FAIL — {err}")
        elif "hit_rate" in ds_data:
            hr = ds_data["hit_rate"] * 100
            rt = ds_data.get("runtime", 0)
            lines.append(f"  {ds_name} ({size_tag}): hit={hr:.1f}%  time={rt:.1f}s")
        else:
            lines.append(f"  {ds_name} ({size_tag}): no result")

    return "\n".join(lines)


def _error_result(error_code, eval_time, error_msg):
    from openevolve.evaluation_result import EvaluationResult
    return EvaluationResult(
        metrics={
            "combined_score": float(error_code),
            "rl_normalized_reward": float(error_code),
            "validity": 0.0,
            "eval_time": eval_time,
        },
        artifacts={"error": error_msg},
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = evaluate(sys.argv[1])
        print(f"Score: {result.metrics.get('combined_score')}")
        print(f"Hit rate: {result.metrics.get('objective_value')}")
        if result.artifacts.get("error"):
            print(f"Error: {result.artifacts['error']}")
    else:
        print("Usage: python evaluator_modular.py <program_path>")
