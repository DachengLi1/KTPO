"""
Cloudcast Modular Evaluator — subprocess-based for GIL-free parallelism.

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
PER_CONFIG_TIMEOUT = int(os.environ.get("LEARNX_CLOUDCAST_CFG_TIMEOUT", "60"))
SUBPROCESS_TIMEOUT = int(os.environ.get("LEARNX_CLOUDCAST_EVAL_TIMEOUT", "120"))

CONFIG_INFO = {
    "intra_aws":   {"type": "intra-cloud", "provider": "AWS",   "dsts": 6, "partitions": 10},
    "intra_azure": {"type": "intra-cloud", "provider": "Azure", "dsts": 6, "partitions": 10},
    "intra_gcp":   {"type": "intra-cloud", "provider": "GCP",   "dsts": 5, "partitions": 10},
    "inter_agz":   {"type": "inter-cloud", "provider": "AWS/GCP/Azure", "dsts": 6, "partitions": 10},
    "inter_gaz2":  {"type": "inter-cloud", "provider": "GCP/AWS/Azure", "dsts": 7, "partitions": 10},
}


def evaluate(program_path, temp_dir=None):
    """Evaluate cloudcast program by spawning verify.py as a subprocess."""
    from openevolve.evaluation_result import EvaluationResult
    from openevolve.modular_utils.error_constants import ErrorCodes
    from openevolve.modular_utils.score_transform import (
        create_score_transform_config_from_dict,
        transform_score_for_rl,
    )
    from openevolve.modular_utils.yaml_config_service import load_yaml_variables

    yaml_vars = load_yaml_variables()
    num_vms = 2
    if isinstance(yaml_vars, dict):
        core_params = yaml_vars.get("core_parameters")
        if isinstance(core_params, dict) and core_params.get("num_vms") is not None:
            try:
                num_vms = int(core_params["num_vms"])
            except Exception:
                num_vms = 2

    fd, output_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    start_time = time.time()

    try:
        proc = subprocess.run(
            [
                sys.executable, VERIFY_SCRIPT,
                "--program", program_path,
                "--output", output_path,
                "--timeout", str(PER_CONFIG_TIMEOUT),
                "--num-vms", str(num_vms),
            ],
            timeout=SUBPROCESS_TIMEOUT,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        eval_time = time.time() - start_time

        try:
            with open(output_path) as f:
                data = json.load(f)
        except Exception:
            stderr_tail = (proc.stderr or "")[-2000:]
            return _error_result(
                ErrorCodes.EXECUTION_ERROR, eval_time,
                f"Failed to parse evaluation output. exit={proc.returncode} stderr: {stderr_tail}",
            )

        if "error" in data and data.get("combined_score", 0.0) == 0.0 and "per_config" not in data:
            return _error_result(
                ErrorCodes.EXECUTION_ERROR, eval_time, data["error"],
            )

        combined_score = data.get("combined_score", 0.0)
        total_cost = data.get("total_cost", 0.0)
        total_transfer_time = data.get("total_transfer_time", 0.0)
        num_configs = data.get("num_configs", 5)
        num_ok = data.get("num_succeeded", 0)

        score_transform_cfg = create_score_transform_config_from_dict(yaml_vars)
        rl_reward = transform_score_for_rl(combined_score, score_transform_cfg)

        extra = {}
        for cfg_name, cfg_data in data.get("per_config", {}).items():
            if isinstance(cfg_data, dict) and "cost" in cfg_data:
                extra[f"cost_{cfg_name}"] = cfg_data["cost"]
                extra[f"runtime_{cfg_name}_s"] = cfg_data.get("runtime", 0.0)

        metrics = {
            "combined_score": combined_score,
            "rl_normalized_reward": rl_reward,
            "score": combined_score * 100.0,
            "validity": float(num_ok) / num_configs if num_configs > 0 else 0.0,
            "total_cost": total_cost,
            "total_transfer_time": total_transfer_time,
            "eval_time": eval_time,
            **extra,
        }

        artifacts = {}
        feedback = _format_per_config_feedback(data)
        if feedback:
            artifacts["error"] = feedback

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


def _format_per_config_feedback(data):
    """Build a concise per-config report for multistep feedback."""
    per_cfg = data.get("per_config", {})
    if not per_cfg:
        return None

    lines = [
        f"Evaluated on 5 network configs (per-config timeout={PER_CONFIG_TIMEOUT}s). "
        "Do NOT hardcode config-specific logic; final evaluation may use different configs:"
    ]

    for cfg_name in ["intra_aws", "intra_azure", "intra_gcp", "inter_agz", "inter_gaz2"]:
        cfg_data = per_cfg.get(cfg_name, {})
        info = CONFIG_INFO.get(cfg_name, {})
        tag = f"{info.get('type', '?')}, {info.get('provider', '?')}, {info.get('dsts', '?')} dsts, {info.get('partitions', '?')} partitions"

        if "error" in cfg_data:
            err = cfg_data["error"]
            if len(err) > 120:
                err = err[:120] + "..."
            lines.append(f"  {cfg_name} ({tag}): FAIL — {err}")
        elif "cost" in cfg_data:
            cost = cfg_data["cost"]
            rt = cfg_data.get("runtime", 0)
            lines.append(f"  {cfg_name} ({tag}): cost=${cost:.2f}  algo_time={rt:.2f}s")
        else:
            lines.append(f"  {cfg_name} ({tag}): no result")

    total_cost = data.get("total_cost", 0.0)
    score = data.get("combined_score", 0.0)
    lines.append(f"  TOTAL: cost=${total_cost:.2f}  normalized_score={score:.4f}")

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
        print(f"Cost:  {result.metrics.get('total_cost')}")
        if result.artifacts.get("error"):
            print(f"Feedback:\n{result.artifacts['error']}")
    else:
        print("Usage: python evaluator_modular.py <program_path>")
