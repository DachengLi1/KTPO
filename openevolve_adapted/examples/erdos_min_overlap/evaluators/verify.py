"""
Erdős Minimum Overlap Problem — Validation & Scoring

Goal: minimize C₅ = max_k ∫ h(x)(1 - h(x+k)) dx
      for a step function h: [0,2] → [0,1] with ∫h(x)dx = 1.

Discretization:
  h is an array of n_points samples over [0, 2].
  dx = 2.0 / n_points.
  Constraints: 0 ≤ h[i] ≤ 1 for all i; sum(h) == n_points / 2 exactly.
  C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

This is a MINIMIZATION problem: lower C₅ is better.
"""

import numpy as np
from typing import Tuple, Any, Optional

from openevolve.modular_utils.evaluation_controller import get_current_problem_config

PROBLEM_CONFIG = get_current_problem_config()

# Accepted keys for extracting h-values from solution dict
ACCEPTED_KEYS = ("h_values", "h", "heights", "sequence")

DEFAULT_ERROR_RETURN = 1e8
MAX_N_POINTS = 50000
MIN_N_POINTS = 10


def _coerce_to_1d_array(solution_data: Any) -> Tuple[Optional[np.ndarray], str]:
    """
    Extract a 1D float array of h-values from accepted formats.
    Supports top-level keys and nested under 'current_solution'.
    """
    arr = None

    if isinstance(solution_data, dict):
        # 1) try top-level accepted keys
        for k in ACCEPTED_KEYS:
            if k in solution_data:
                arr = solution_data[k]
                break

        # 2) if not found, try nested under 'current_solution'
        if arr is None and isinstance(solution_data.get("current_solution"), dict):
            cs = solution_data["current_solution"]
            for k in ACCEPTED_KEYS + ("data",):
                if k in cs:
                    arr = cs[k]
                    break
        if arr is None:
            return None, (
                "Dictionary input must contain one of keys: "
                f"{', '.join(ACCEPTED_KEYS)} (optionally nested under 'current_solution')"
            )
    else:
        # direct array-like
        arr = solution_data

    # Coerce to numpy array and validate
    try:
        arr = np.asarray(arr, dtype=np.float64)
    except Exception as e:
        return None, f"Could not convert input to float array: {e}"

    if arr.ndim != 1 or arr.size == 0:
        return None, f"h_values must be a non-empty 1D array, got shape {arr.shape}"

    return arr, ""


def validate_solution(solution_data: Any) -> Tuple[bool, str]:
    """
    Validate an Erdős minimum overlap solution.

    Returns:
        (is_valid, error_message)
    """
    try:
        h_values, err = _coerce_to_1d_array(solution_data)
        if h_values is None:
            return False, err

        n_points = h_values.size

        # Length bounds
        if n_points < MIN_N_POINTS:
            return False, f"Too few points: {n_points} < {MIN_N_POINTS}"
        if n_points > MAX_N_POINTS:
            return False, f"Too many points: {n_points} > {MAX_N_POINTS}"

        # Finite check
        if not np.all(np.isfinite(h_values)):
            return False, "h_values contain NaN or infinite values"

        # Range check: h(x) ∈ [0, 1]
        if np.any(h_values < -1e-8) or np.any(h_values > 1.0 + 1e-8):
            return False, (
                f"h(x) must be in [0, 1]. "
                f"Range: [{float(h_values.min()):.6f}, {float(h_values.max()):.6f}]"
            )

        # Integral check: sum(h) should equal n_points / 2
        # We allow normalization (same as TTT-Discover), so just check sum > 0
        total = float(np.sum(h_values))
        if total < 1e-8:
            return False, "Sum of h_values is effectively zero; invalid"

        return True, "Valid Erdős step function"

    except Exception as e:
        return False, f"Validation error: {e}"


def compute_objective_value(solution_data: Any) -> float:
    """
    Compute the C₅ overlap constant from validated h-values.

    This independently recomputes C₅ (anti-cheat) and applies normalization
    if the sum constraint is not exactly satisfied.

    Returns:
        C₅ value (float). Lower is better.
        Returns DEFAULT_ERROR_RETURN on any failure.
    """
    try:
        is_valid, msg = validate_solution(solution_data)
        if not is_valid:
            return DEFAULT_ERROR_RETURN

        h_values, _ = _coerce_to_1d_array(solution_data)
        n_points = h_values.size

        # Normalize: enforce sum(h) == n_points / 2
        target_sum = n_points / 2.0
        current_sum = float(np.sum(h_values))
        if current_sum != target_sum:
            h_values = h_values * (target_sum / current_sum)
            # Re-check range after normalization
            if np.any(h_values < -1e-8) or np.any(h_values > 1.0 + 1e-8):
                return DEFAULT_ERROR_RETURN

        # Compute C₅ = max of cross-correlation of h and (1-h)
        dx = 2.0 / n_points
        j_values = 1.0 - h_values
        correlation = np.correlate(h_values, j_values, mode="full") * dx
        c5 = float(np.max(correlation))

        if not np.isfinite(c5):
            return DEFAULT_ERROR_RETURN

        return c5

    except Exception as e:
        print(f"Error computing C5 objective: {e}")
        return DEFAULT_ERROR_RETURN


if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON solution data")
    args = parser.parse_args()

    try:
        with open(args.input, "r") as f:
            solution_data = json.load(f)
    except Exception as e:
        print(f"INVALID: failed to load JSON: {e}")
        sys.exit(0)

    is_valid, msg = validate_solution(solution_data)
    if not is_valid:
        print(f"INVALID: {msg}")
        sys.exit(0)

    value = compute_objective_value(solution_data)
    print(f"{value}")
