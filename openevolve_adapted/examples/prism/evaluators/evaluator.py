import copy
import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
from dataclasses import dataclass

GPU_MEM_SIZE = 80 # GB
MIN_INT = float('-inf')  # Define MIN_INT as negative infinity

@dataclass
class Model:
    model_name: str
    model_size: int
    req_rate: int
    slo: int
    cur_gpu_id: int


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """
    Run a function with a timeout using concurrent.futures
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def safe_float(value):
    """Convert a value to float safely"""
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def verify_gpu_mem_constraint(placement_data: dict[int, list[Model]]) -> bool:
    """
    Verify the whether models can fit into GPU memory
    """
    # Check if the placement data is valid
    if placement_data is None:
        return False

    # Check if the placement data is valid
    for gpu_id, models in placement_data.items():
        if sum(model.model_size for model in models) > GPU_MEM_SIZE:
            return False

    return True


def calculate_kvcache_pressure(placement_data: dict[int, list[Model]]) -> float:
    """
    Calculate the KVCache pressure
    """
    max_kvpr = MIN_INT
    for gpu_id, models in placement_data.items():
        total_model_size = sum(model.model_size for model in models)
        total_weighted_req_rate = sum(model.req_rate / model.slo for model in models)
        if GPU_MEM_SIZE - total_model_size > 0:
            kvpr = total_weighted_req_rate / (GPU_MEM_SIZE - total_model_size)
        else:
            kvpr = 1000000
        max_kvpr = max(max_kvpr, kvpr)

    return max_kvpr


def generate_test_gpu_models(num_tests=50):
    """
    Generate multiple test signals with different characteristics
    """
    test_cases = []
    np.random.seed(42)

    for i in range(num_tests):
        gpu_num = np.random.randint(5, 10)
        gpu_models = []
        for j in range(gpu_num*2):
            model_size = np.random.randint(10, 30)
            req_rate = np.random.randint(1, 10)
            slo = np.random.randint(5, 10)
            gpu_models.append(Model(model_name=f"model_{j}", model_size=model_size, req_rate=req_rate, slo=slo, cur_gpu_id=j))

        test_cases.append((gpu_num, gpu_models))

    return test_cases

def evaluate(program_path):
    """
    Main evaluation function that tests the signal processing algorithm
    on multiple test signals and calculates the composite performance metric.
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if required function exists
        if not hasattr(program, "compute_model_placement"):
            return {
                "max_kvpr": 0.0,
                "success_rate": 0.0,
                "combined_score": 0.0,
                "error": "Missing compute_model_placement function",
                }

        # Generate test gpu and models
        test_gpu_models = generate_test_gpu_models()

        # Collect metrics across all tests
        all_kvpr = []
        all_metrics = []
        successful_runs = 0

        for i, (gpu_num, gpu_models) in enumerate(test_gpu_models):
            try:
                # Deep copy models so program cannot mutate originals.
                # Program gets copies; we rebuild result using originals for scoring.
                model_copies = copy.deepcopy(gpu_models)
                name_to_original = {m.model_name: m for m in gpu_models}

                # Run the algorithm with timeout (on COPIES)
                start_time = time.time()
                raw_result = run_with_timeout(
                    program.compute_model_placement,
                    kwargs={
                        'gpu_num': gpu_num,
                        'models': model_copies
                    },
                    timeout_seconds=10
                )
                execution_time = time.time() - start_time

                # Rebuild result using original (untampered) model objects
                if not isinstance(raw_result, dict):
                    result = raw_result  # will fail validation below
                else:
                    result = {}
                    remap_ok = True
                    for gpu_id, assigned in raw_result.items():
                        if not isinstance(assigned, list):
                            remap_ok = False
                            break
                        remapped = []
                        for m in assigned:
                            orig = name_to_original.get(getattr(m, 'model_name', None))
                            if orig is None:
                                remap_ok = False
                                break
                            remapped.append(orig)
                        if not remap_ok:
                            break
                        result[gpu_id] = remapped
                    if not remap_ok:
                        return {
                            "max_kvpr": 0.0,
                            "success_rate": 0.0,
                            "combined_score": 0.0,
                            "error": f"Placement {i}: could not remap models back to originals",
                        }

                # Validate result format
                if not isinstance(result, dict):
                    return {
                        "max_kvpr": 0.0,
                        "success_rate": 0.0,
                        "combined_score": 0.0,
                        "error": f"Placement {i}: Expected dict, got {type(result).__name__}",
                    }

                # Validate: cannot use more GPUs than available
                if len(result) > gpu_num:
                    return {
                        "max_kvpr": 0.0,
                        "success_rate": 0.0,
                        "combined_score": 0.0,
                        "error": f"Placement {i}: used {len(result)} GPUs but only {gpu_num} available",
                    }

                # Validate all models are placed
                placed_models = []
                for gpu_id, assigned_models in result.items():
                    if not isinstance(assigned_models, list):
                        return {
                            "max_kvpr": 0.0,
                            "success_rate": 0.0,
                            "combined_score": 0.0,
                            "error": f"GPU {gpu_id} value must be list, got {type(assigned_models).__name__}",
                        }
                    placed_models.extend(assigned_models)

                if len(placed_models) != len(gpu_models):
                    return {
                        "max_kvpr": 0.0,
                        "success_rate": 0.0,
                        "combined_score": 0.0,
                        "error": f"Not all models placed: {len(placed_models)}/{len(gpu_models)}",
                    }

                # Check for duplicate placements (by object identity)
                placed_ids = [id(m) for m in placed_models]
                if len(set(placed_ids)) != len(placed_ids):
                    return {
                        "max_kvpr": 0.0,
                        "success_rate": 0.0,
                        "combined_score": 0.0,
                        "error": f"Duplicate models detected",
                    }

                # Check placed models are the exact input objects
                original_ids = {id(m) for m in gpu_models}
                if set(placed_ids) != original_ids:
                    return {
                        "max_kvpr": 0.0,
                        "success_rate": 0.0,
                        "combined_score": 0.0,
                        "error": "Placed models don't match input models (missing or foreign models)",
                    }

                # Verify GPU memory constraints
                if not verify_gpu_mem_constraint(result):
                    return {
                        "max_kvpr": 0.0,
                        "success_rate": 0.0,
                        "combined_score": 0.0,
                        "error": f"GPU memory constraint violated",
                    }

                # Calculate metrics using the generated test signal
                max_kvpr = calculate_kvcache_pressure(result)

                # Validate: kvpr must be finite and positive
                if max_kvpr <= 0 or np.isinf(max_kvpr) or np.isnan(max_kvpr):
                    print(f"Placement {i}: Invalid kvpr={max_kvpr}")
                    continue

                # Store metrics
                metrics = {
                    'max_kvpr': safe_float(max_kvpr),
                    'execution_time': safe_float(execution_time),
                }

                all_kvpr.append(safe_float(max_kvpr))
                all_metrics.append(metrics)
                successful_runs += 1

            except TimeoutError:
                print(f"Placement {i}: Timeout")
                continue
            except Exception as e:
                print(f"Placement {i}: Error - {str(e)}")
                continue

        # If no successful runs, return minimal scores
        if successful_runs == 0:
            return {
                    "max_kvpr": 0.0,
                    "success_rate": 0.0,
                    "combined_score": 0.0,
                    "error": "All test signals failed"
                }

        print(all_metrics)
        # Calculate aggregate metrics
        avg_kvpr = np.mean(all_kvpr)
        if avg_kvpr != 0:
            avg_kvpr = 1.0 / avg_kvpr
        avg_execution_time = np.mean([m['execution_time'] for m in all_metrics])
        success_rate = successful_runs / len(test_gpu_models)

        # Any failed test case → score 0 (confirmed by task author)
        if successful_runs < len(test_gpu_models):
            return {
                    "max_kvpr": 0.0,
                    "execution_time": safe_float(avg_execution_time) if all_metrics else 0.0,
                    "success_rate": safe_float(success_rate),
                    "combined_score": 0.0,
                    "error": f"Only {successful_runs}/{len(test_gpu_models)} test cases passed",
                }

        return {
                "max_kvpr": safe_float(avg_kvpr),
                "execution_time": safe_float(avg_execution_time),
                "success_rate": safe_float(success_rate),
                "combined_score": safe_float(avg_kvpr) + safe_float(success_rate),
            }

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print(traceback.format_exc())
        return {
                "max_kvpr": 0.0,
                "success_rate": 0.0,
                "combined_score": 0.0,
                "error": str(e)
            }


if __name__ == "__main__":
    # Backwards-compat: bridges old evaluate() -> dict to the container JSON
    # protocol.  wrapper.py is auto-injected at build time from
    # skydiscover/evaluation/wrapper.py.
    from wrapper import run

    run(evaluate)
