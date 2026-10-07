import math
from typing import Dict, Any, Optional

import os
from openevolve.modular_utils.error_constants import FORMAT_ERRORS, EXECUTION_ERRORS, ALL_ERROR_CODES

# Globally registered gym (same process as rollout manager)
_GYM = None


def get_database_worst_score() -> Optional[float]:
    """Return the worst combined_score in the current database, or None if unavailable.

    Used by adaptive-k to decide whether a generated program would contribute
    to the gene pool (score > worst) or be immediately evicted (score <= worst).
    """
    if _GYM is None:
        return None
    try:
        return _GYM.database.get_worst_score()
    except Exception:
        return None


def get_database_best_score() -> Optional[float]:
    """Return the best combined_score in the current database, or None."""
    if _GYM is None:
        return None
    try:
        best = _GYM.database.get_best_program()
        if best and best.metrics:
            return best.metrics.get("combined_score", None)
        return None
    except Exception:
        return None


def get_database_size() -> int:
    """Return the current number of programs in the database."""
    if _GYM is None:
        return 0
    try:
        return len(_GYM.database.programs)
    except Exception:
        return 0


def _get_combined_score(metrics: Dict[str, Any], default_low_score: float) -> float:
    """Get combined_score from metrics with fallback."""
    return metrics.get("combined_score", default_low_score)


def _get_rl_normalized_reward(metrics: Dict[str, Any], default_low_score: float) -> float:
    """Get rl_normalized_reward from metrics, fallback to combined_score/default_low_score for error/timeout cases."""
    reward_val = metrics.get("rl_normalized_reward")
    if reward_val is None:
        reward_val = default_low_score
        
    return reward_val


def _get_parent_combined_score(parent_program, default_low_score: float) -> float:
    """Safely get parent program's combined_score."""
    if parent_program is None:
        return default_low_score

    if hasattr(parent_program, 'metrics') and parent_program.metrics:
        return parent_program.metrics.get("combined_score", default_low_score)
    else:
        import logging
        logging.warning(f"parent_program.metrics is None or empty, id={getattr(parent_program, 'id', None)}, falling back to default_low_score={default_low_score}")
        return default_low_score



def _process_reward(metrics: Dict[str, Any], parent_program, reward_process_type: str, default_low_score: float) -> float:
    """Process reward value based on reward processing type.

    Args:
        metrics: Child program evaluation metrics
        parent_program: Parent program object
        reward_process_type: Processing mode string
        default_low_score: Default fallback score

    Returns:
        Processed reward value
    """

    if reward_process_type == "original_reward":
        # Return raw combined_score without any transformation
        return _get_combined_score(metrics, default_low_score)

    elif reward_process_type == "improve_reward":
        # Improvement reward (single step): 1.0 if child > parent, -0.5 if the child is invalid, else 0.0
        child_combined_score = _get_combined_score(metrics, default_low_score)

        if child_combined_score in ALL_ERROR_CODES:
            return -0.5

        parent_combined_score = _get_parent_combined_score(parent_program, default_low_score)

        eps = 1e-6
        if child_combined_score > parent_combined_score + eps:
            return 1.0
        else:
            return 0.0

    else:
        # All other types use rl_normalized_reward as base
        reward_val = _get_rl_normalized_reward(metrics, default_low_score)

        if reward_process_type == "rl_normalized_reward":
            return reward_val
        elif reward_process_type == "format_reward":
            return reward_val if reward_val in FORMAT_ERRORS else 1.0
        elif reward_process_type == "validation_reward":
            return reward_val if reward_val in ALL_ERROR_CODES else 1.0
        else:
            # Fallback for unknown types
            return reward_val


def _sanitize_reward(reward_val: float, default_low_score: float, clip_min: float = -10.0, clip_max: float = 10.0) -> float:
    """Sanitize reward value: handle NaN/Inf and clip to safe range."""
    if math.isnan(reward_val) or math.isinf(reward_val):
        print(f"[WARNING] Invalid reward {reward_val}, using {default_low_score}")
        return default_low_score

    clipped = max(clip_min, min(clip_max, reward_val))
    if abs(reward_val - clipped) > 1e-6:
        print(f"[WARNING] Reward clipped: {reward_val:.2e} -> {clipped:.2f}")
    return clipped


def set_gym(gym):
    global _GYM
    _GYM = gym


async def evolving_gym_rm(args, sample) -> Dict[str, Any]:
    """
    Score LLM output using gym and return reward dictionary.
    Convention:
      - Return dict contains at least args.reward_key with scalar reward (e.g., combined_score)
      - Include metrics (complete evaluator stats) and child_id as auxiliary fields
      - Cache child_program for visualization/analysis
    """
    default_low_score = -1.0
    if _GYM is None:
        return {getattr(args, "reward_key", "reward"): default_low_score, "error": "gym_not_initialized"}

    metadata = sample.metadata or {}
    if isinstance(metadata, str):
        # Safety check: some upstream code may serialize to JSON string, don't handle here, use dict default
        assert False, "metadata should not be str"
        metadata = {}

    parent_program = metadata.get("parent_program", None)
    if not parent_program:
        return {getattr(args, "reward_key", "reward"): default_low_score, "error": "missing_parent_program"}

    # N-repeat evaluation: check metadata first, then env var
    n_repeat = metadata.get("_eval_n_repeat", None)
    if n_repeat is None:
        n_repeat = int(os.environ.get("LEARNX_EVAL_N_REPEAT", "1"))

    try:
        # Async evaluation (with semaphore + timeout internally)
        result = await _GYM.response_scorer(
            sample.response or "", parent_program,
            n_repeat=n_repeat,
        )

    except Exception as e:
        print(f"exception in evolving_gym_rm: {e}")
        return {getattr(args, "reward_key", "reward"): default_low_score, "error": f"exception:{str(e)[:200]}"}

    if result is None or not getattr(result, "child_metrics", None):
        print(f"result is none or child_metrics is none")
        return {getattr(args, "reward_key", "reward"): default_low_score}

    metrics = result.child_metrics or {}
    reward_val = _process_reward(metrics, parent_program, _GYM.reward_process_type, default_low_score)
    reward_val = _sanitize_reward(reward_val, default_low_score)

    # Add child program directly to the main database.
    # This is safe because database.add() is thread-safe, and the parent-selection
    # sync barrier is between rollouts (not within), so concurrent adds during a
    # rollout are fine. Direct add keeps get_worst_score() up-to-date in real time,
    # which is needed for accurate adaptive-k thresholds.
    # target_island: child goes to parent's island (preserves reproductive isolation).
    parent_island = parent_program.metadata.get("island", 0) if (parent_program and hasattr(parent_program, 'metadata') and parent_program.metadata) else None
    orig_parent_score = metadata.get("_original_parent_score", None)
    try:
        if getattr(result, "child_program", None) is not None:
            # When improving-only mode (LEARNX_ACDS_IMPROVING_ONLY=1) is on, only add children that beat their parent
            acds_improving_only = int(os.environ.get("LEARNX_ACDS_IMPROVING_ONLY", "0"))
            should_add = True
            if acds_improving_only:
                child_score = _get_combined_score(metrics, default_low_score)
                parent_score_val = _get_parent_combined_score(parent_program, default_low_score)
                should_add = child_score > parent_score_val
            if should_add:
                _GYM.database.add(result.child_program, iteration=len(_GYM.database.programs),
                                  target_island=parent_island,
                                  parent_score=orig_parent_score)
    except Exception as e:
        print(f"Failed to add to database: {e}")

    # NOTE: B-adaptive sampling recording moved to group-level in sglang_rollout.py
    # (record_group_beat is called once per parent group, not per child)

    out = {
        getattr(args, "reward_key", "reward"): reward_val,
        "metrics": metrics,
        "child_id": getattr(getattr(result, "child_program", None), "id", None),
        "parent_island": parent_island if parent_island is not None else 0,
    }
    if getattr(result, "iteration_time", None) is not None:
        out["iteration_time"] = result.iteration_time
    if getattr(result, "artifacts", None):
        out["artifacts"] = result.artifacts
    if getattr(result, "runtime_environment_path", None):
        out["runtime_environment_path"] = result.runtime_environment_path
    # Expose child_program for multi-step refinement (need code + metrics for next step's parent)
    if getattr(result, "child_program", None) is not None:
        out["child_program"] = result.child_program

    return out


# ---------------------------------------------------------------------------
# Multi-step accumulated reward
# ---------------------------------------------------------------------------

_REWARD_AGG_LOGGED = False

def compute_multistep_reward(
    raw_scores: list[float],
    agg: str = "sum",
    validities: list[bool] | None = None,
) -> float:
    """Compute multi-step refinement reward.

    raw_scores[0] is the parent score (excluded from aggregation).
    Each step's score includes evaluator error codes (-0.1..-0.4) as-is.

    Aggregation modes:
        'sum': R = sum(raw_scores[1:])  — ThetaEvolve style, penalizes every bad step.
               k=1: R = s_1.  k=4: R = s_1 + s_2 + s_3 + s_4
        'mean': R = mean(raw_scores[1:]) — normalized per step count, stable across k values.
               k=1: R = s_1.  k=4: R = (s_1 + s_2 + s_3 + s_4) / 4
        'max': R = max(raw_scores[1:])  — rewards best step, encourages exploration.
               k=1: R = s_1.  k=4: R = max(s_1, s_2, s_3, s_4)
        'max_improve': KTPO improvement reward using the best of the k steps.
               +1.0 if max(raw_scores[1:]) > parent; -0.5 only if ALL k steps are
               invalid; otherwise 0.0 (at least one valid step, none beats the parent).

    Args:
        raw_scores: [parent, step1, step2, ...] raw evaluator scores.
        agg: Aggregation function ('sum', 'mean', 'max', or 'max_improve').
        validities: [parent_valid, step1_valid, ...] per-step validity flags.
                    Required for 'max_improve'; ignored by other modes.
    """
    global _REWARD_AGG_LOGGED
    if not _REWARD_AGG_LOGGED:
        print(f"[compute_multistep_reward] agg='{agg}', k={len(raw_scores)-1} "
              f"(will use {agg}(raw_scores[1:]) for all subsequent calls)", flush=True)
        _REWARD_AGG_LOGGED = True

    if len(raw_scores) < 2:
        return 0.0
    step_scores = raw_scores[1:]

    if agg == "max_improve":
        parent_score = raw_scores[0]
        step_valids = validities[1:] if validities and len(validities) >= 2 else [True] * len(step_scores)
        if not any(step_valids):
            return -0.5
        best_child = max(step_scores)
        if best_child > parent_score:
            return 1.0
        return 0.0
    elif agg == "max":
        result = max(step_scores)
    elif agg == "mean":
        result = sum(step_scores) / len(step_scores)
    else:
        result = sum(step_scores)
    return result
