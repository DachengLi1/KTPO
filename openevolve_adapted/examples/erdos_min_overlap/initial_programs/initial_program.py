# EVOLVE-BLOCK-START
"""
Erdős Minimum Overlap Problem — Search for step function h minimizing C₅.

C₅ = max_k ∫ h(x)(1 - h(x+k)) dx

Constraints:
  - h: [0,2] → [0,1]  (discretized as n_points samples)
  - ∫₀² h(x) dx = 1   (i.e., sum(h) == n_points / 2)

Lower C₅ is better. Current record: C₅ ≈ 0.38092.
"""
import numpy as np
import time


def compute_c5(h_values):
    """Compute C₅ overlap for a given step function h."""
    n = len(h_values)
    dx = 2.0 / n
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    return float(np.max(correlation))


def search_for_best_h():
    """Search for h-values minimizing C₅ using random perturbations."""
    rng = np.random.default_rng(42)
    best_h = None
    best_c5 = np.inf
    start_time = time.time()

    for trial in range(200):
        if time.time() - start_time > MAX_TIME * 0.9:
            break

        # Random initialization: h = 0.5 + perturbation (matching TTT-Discover)
        n_points = rng.integers(40, 100)
        h = np.ones(n_points) * 0.5
        perturbation = rng.uniform(-0.4, 0.4, n_points)
        perturbation = perturbation - np.mean(perturbation)
        h = h + perturbation
        h = np.clip(h, 0.0, 1.0)

        # Normalize so sum(h) == n_points / 2
        target = n_points / 2.0
        s = np.sum(h)
        if s > 1e-12:
            h = h * (target / s)
        h = np.clip(h, 0.0, 1.0)

        c5 = compute_c5(h)
        if c5 < best_c5:
            best_c5 = c5
            best_h = h.copy()
            save_checkpoint(best_h, best_c5, time.time() - start_time)

    return best_h


# EVOLVE-BLOCK-END


def save_checkpoint(h_values, c5_value, elapsed):
    """Save current best h-values to file."""
    try:
        current_solution = {'data': h_values.tolist()}
        save_search_results(
            best_perfect_solution=None,
            current_solution=current_solution,
            target_value=TARGET_VALUE
        )
        print(f"[CHECKPOINT] t={elapsed:.1f}s, C5={c5_value:.6f}, n={len(h_values)}")
    except Exception as e:
        print(f"[WARNING] Failed to save: {e}")


def run_erdos_optimization():
    """Main entry point for Erdős optimization."""
    h_values = search_for_best_h()
    c5_value = compute_c5(h_values)
    print(f"Generated h-function: C5={c5_value:.6f}, n_points={len(h_values)}")
    save_checkpoint(h_values, c5_value, MAX_TIME)
    return h_values, c5_value


if __name__ == "__main__":

    ######## get parameters from config ########
    from openevolve.modular_utils.file_io_controller import save_search_results
    from openevolve.modular_utils.evaluation_controller import get_current_problem_config
    PROBLEM_CONFIG = get_current_problem_config()
    TARGET_VALUE = PROBLEM_CONFIG['core_parameters']['target_value']
    PROBLEM_TYPE = PROBLEM_CONFIG['problem_type']
    MAX_TIME = PROBLEM_CONFIG['time_limit'] * 0.9  # use 90% of the allowed time
    print(f"MAX_TIME from config: {MAX_TIME} seconds")
    ###############################################

    h_values, c5_value = run_erdos_optimization()
    print(f"\nGenerated {PROBLEM_TYPE} function:")
    print(f"C5 value: {c5_value:.6f}")
    print(f"Target: {TARGET_VALUE}")
    print(f"Function length: {len(h_values)}")
