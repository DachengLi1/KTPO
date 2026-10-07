import sys
import os
sys.path.append(os.path.dirname(__file__))
from openevolve.modular_utils.evaluation_controller import create_evaluation_controller, ObjectiveBasedProblemEvaluator, get_current_problem_config
from verify import validate_solution, compute_objective_value

# Get problem configuration from YAML
PROBLEM_CONFIG = get_current_problem_config()


def evaluate(program_path, temp_dir=None):
    """
    Modular problem evaluation using universal controller

    Note: Erdős minimum overlap is a MINIMIZATION problem (lower C₅ is better).
    VanillaObjectiveScoring with maximize=False will use 1/(c5 + eps) as combined_score,
    so higher combined_score = better (smaller C₅).
    """
    problem_evaluator = ObjectiveBasedProblemEvaluator(
        validate_func=validate_solution,
        compute_objective_func=compute_objective_value,
        maximize=False  # MINIMIZATION: lower C₅ values are better
    )

    evaluation_controller = create_evaluation_controller(problem_evaluator, PROBLEM_CONFIG)

    return evaluation_controller.evaluate(program_path, temp_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = evaluate(sys.argv[1])
        print(f"Score: {result.metrics.get('combined_score')}")
    else:
        print("Usage: python evaluator_modular.py <program_path>")
