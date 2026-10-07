import torch
from slime.utils.types import Sample


__all__ = ["check_reward_nonzero_std", "check_evolving_gym_acds_trainable", "check_evolving_gym_beat_nonzero"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    return torch.tensor(rewards, dtype=torch.float64).std() > 1e-6


def check_evolving_gym_acds_trainable(args, samples: list[Sample], **kwargs):
    """Record the parent's improvement rate for the pool update; does NOT filter.

    Hooked in as the dynamic-sampling "filter" only so that it sees every
    generated group. For the group's parent it computes the improvement rate
    (fraction of the group's k-step trajectories whose best step beats the
    parent) and passes it to ``database.classify_parent`` (used by
    improvement-rate eviction). It always returns True, so every group is kept
    for training.
    """
    gym = getattr(args, '_acds_gym', None)
    valid_only = getattr(args, "evolving_gym_acds_valid_only_beat_rate", False)

    # Compute improvement rate B
    # Use max(mrs[1:]) when reward_agg="max" so beat detection matches the
    # actual reward signal (best score across all k refinement steps).
    reward_agg = getattr(args, "multistep_reward_agg", "sum")
    parent_score = None
    parent_id = None
    n_beat = 0
    n_total = 0
    n_valid = 0
    for sample in samples:
        r = sample.reward
        if not isinstance(r, dict):
            continue
        mrs = r.get("multistep_raw_scores")
        if not (mrs and len(mrs) >= 2):
            continue
        p_score = float(mrs[0])
        if reward_agg in ("max", "max_improve"):
            c_score = max(float(x) for x in mrs[1:])
        else:
            c_score = float(mrs[-1])
        if parent_score is None:
            parent_score = p_score
            parent_id = r.get("original_parent_id")
        n_total += 1
        # Check validity: child has positive score (valid program)
        mvs = r.get("multistep_validities")
        child_valid = any(mvs[1:]) if isinstance(mvs, list) and len(mvs) >= 2 else (c_score > 0)
        if child_valid:
            n_valid += 1
        if c_score > p_score:
            n_beat += 1

    if valid_only and n_valid > 0:
        beat_rate = n_beat / n_valid
    else:
        beat_rate = (n_beat / n_total) if n_total > 0 else 0.0

    # Determine parent island from reward metadata
    parent_island = -1
    for sample in samples:
        r = sample.reward
        if isinstance(r, dict):
            pi = r.get("parent_island")
            if pi is not None:
                parent_island = int(pi)
                break

    # Record (parent_score, improvement rate); mastered scores feed the eviction threshold
    if gym is not None and gym.database.adaptive_sampling_enabled and parent_id:
        try:
            gym.database.classify_parent(parent_id, beat_rate, parent_score or 0.0,
                                         parent_island=parent_island)
        except Exception as e:
            print(f"[IR-evict] classify_parent error: {e}")

    return True  # accept ALL groups for training


def check_evolving_gym_beat_nonzero(args, samples: list[Sample], **kwargs):
    """Baseline filter (no improvement-rate eviction): accept groups with 0 < B < 1.0."""
    reward_agg = getattr(args, "multistep_reward_agg", "sum")
    n_beat = 0
    n_valid = 0
    for sample in samples:
        r = sample.reward
        if not isinstance(r, dict):
            continue
        mrs = r.get("multistep_raw_scores")
        mvs = r.get("multistep_validities")
        if not (mrs and len(mrs) >= 2):
            continue
        p_score = float(mrs[0])
        if reward_agg in ("max", "max_improve"):
            c_score = max(float(x) for x in mrs[1:])
        else:
            c_score = float(mrs[-1])
        child_valid = any(mvs[1:]) if isinstance(mvs, list) and len(mvs) >= 2 else True
        if child_valid and p_score > 0:
            n_valid += 1
            if c_score > p_score:
                n_beat += 1

    if n_valid == 0:
        return False
    beat_rate = n_beat / n_valid
    return 1e-9 < beat_rate < 1.0 - 1e-9
