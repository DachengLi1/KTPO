#!/usr/bin/env python3
"""
Cloudcast Evaluation Worker — runs as subprocess for GIL-free parallelism.

Usage: python verify.py --program <path> --output <path> [--timeout 60]
"""
import sys
import os
import json
import copy
import time
import math
import signal
import traceback
import importlib.util
import argparse

EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(EVALUATOR_DIR)
sys.path.insert(0, EVALUATOR_DIR)

from broadcast import BroadCastTopology
from simulator import BCSimulator
from utils import make_nx_graph

DATASETS_DIR = os.path.join(TASK_DIR, "datasets")
PROFILES_DIR = os.path.join(DATASETS_DIR, "profiles")
CONFIGS_DIR = os.path.join(DATASETS_DIR, "examples", "config")

LOWER_COST = 1199.00
UPPER_COST = 626.24

CONFIG_FILENAMES = (
    "intra_aws.json",
    "intra_azure.json",
    "intra_gcp.json",
    "inter_agz.json",
    "inter_gaz2.json",
)

CONFIG_LABELS = {
    "intra_aws":    {"type": "intra", "provider": "AWS", "ndst": 6},
    "intra_azure":  {"type": "intra", "provider": "Azure", "ndst": 6},
    "intra_gcp":    {"type": "intra", "provider": "GCP", "ndst": 5},
    "inter_agz":    {"type": "inter", "provider": "AWS→GCP→Azure", "ndst": 6},
    "inter_gaz2":   {"type": "inter", "provider": "GCP→AWS→Azure", "ndst": 7},
}


class _ConfigTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _ConfigTimeout()


def _write(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def validate_broadcast_topology(bc_t, source_node, terminal_nodes, num_partitions, G):
    """
    Validate that the broadcast topology is complete and correct.
    Ported from skydiscover official benchmark evaluator.

    Returns:
        (is_valid, error_message) tuple
    """
    # Check 1: Verify all destinations are present
    if set(bc_t.dsts) != set(terminal_nodes):
        missing_dsts = set(terminal_nodes) - set(bc_t.dsts)
        extra_dsts = set(bc_t.dsts) - set(terminal_nodes)
        return False, f"Destination mismatch: missing={missing_dsts}, extra={extra_dsts}"

    # Check 2: Verify source matches
    if bc_t.src != source_node:
        return False, f"Source mismatch: expected={source_node}, got={bc_t.src}"

    # Check 3: Verify all partitions exist for all destinations
    missing_partitions = []
    empty_partitions = []
    invalid_paths = []

    for dst in terminal_nodes:
        if dst not in bc_t.paths:
            return False, f"Missing destination '{dst}' in paths"

        for partition_id in range(num_partitions):
            partition_key = str(partition_id)

            if partition_key not in bc_t.paths[dst]:
                missing_partitions.append((dst, partition_id))
                continue

            partition_paths = bc_t.paths[dst][partition_key]

            if partition_paths is None or len(partition_paths) == 0:
                empty_partitions.append((dst, partition_id))
                continue

            # Check 4: Verify paths form valid routes from source to destination
            # Also Check 6: Verify edge_data matches original graph (prevents cost tampering)
            path_nodes = [source_node]
            path_valid = True

            for edge in partition_paths:
                if len(edge) < 3:
                    invalid_paths.append((dst, partition_id, "edge format invalid"))
                    path_valid = False
                    break

                edge_src, edge_dst, edge_data = edge[0], edge[1], edge[2]

                if not G.has_edge(edge_src, edge_dst):
                    invalid_paths.append((dst, partition_id, f"edge {edge_src}->{edge_dst} not in graph"))
                    path_valid = False
                    break

                if path_nodes[-1] != edge_src:
                    invalid_paths.append((dst, partition_id, f"path discontinuity: expected {path_nodes[-1]}, got {edge_src}"))
                    path_valid = False
                    break

                # Verify edge_data cost/throughput match the original graph
                if isinstance(edge_data, dict):
                    real_edge = G[edge_src][edge_dst]
                    real_cost = real_edge.get("cost")
                    given_cost = edge_data.get("cost")
                    if real_cost is not None and given_cost is not None:
                        if abs(float(given_cost) - float(real_cost)) > 1e-6:
                            invalid_paths.append((dst, partition_id,
                                f"cost tampered on {edge_src}->{edge_dst}: given={given_cost}, real={real_cost}"))
                            path_valid = False
                            break
                    real_tp = real_edge.get("throughput")
                    given_tp = edge_data.get("throughput")
                    if real_tp is not None and given_tp is not None:
                        if abs(float(given_tp) - float(real_tp)) > 1e-6:
                            invalid_paths.append((dst, partition_id,
                                f"throughput tampered on {edge_src}->{edge_dst}: given={given_tp}, real={real_tp}"))
                            path_valid = False
                            break

                path_nodes.append(edge_dst)

            if path_valid and path_nodes[-1] != dst:
                invalid_paths.append((dst, partition_id, f"path does not reach destination: ends at {path_nodes[-1]}, expected {dst}"))

    errors = []
    if missing_partitions:
        errors.append(f"Missing partitions: {missing_partitions}")
    if empty_partitions:
        errors.append(f"Empty partitions: {empty_partitions}")
    if invalid_paths:
        errors.append(f"Invalid paths: {invalid_paths}")

    if errors:
        return False, "Validation failed: " + "; ".join(errors)

    # Check 5: Verify all data volumes are accounted for
    expected_total_partitions = len(terminal_nodes) * num_partitions
    actual_partitions = 0
    for dst in terminal_nodes:
        for partition_id in range(num_partitions):
            partition_key = str(partition_id)
            if (partition_key in bc_t.paths[dst] and
                    bc_t.paths[dst][partition_key] is not None and
                    len(bc_t.paths[dst][partition_key]) > 0):
                actual_partitions += 1

    if actual_partitions != expected_total_partitions:
        return False, f"Data loss detected: expected {expected_total_partitions} partitions, got {actual_partitions}"

    return True, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-config timeout in seconds")
    parser.add_argument("--num-vms", type=int, default=2)
    args = parser.parse_args()

    per_config_timeout = args.timeout
    num_vms = args.num_vms

    # Build template graph once
    template_graph = make_nx_graph(
        cost_path=os.path.join(PROFILES_DIR, "cost.csv"),
        throughput_path=os.path.join(PROFILES_DIR, "throughput.csv"),
        num_vms=num_vms,
    )

    # Load candidate program
    try:
        spec = importlib.util.spec_from_file_location("candidate", args.program)
        if spec is None or spec.loader is None:
            _write(args.output, {"error": f"Cannot load {args.program}", "combined_score": 0.0})
            return
        prog = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prog)
    except Exception as e:
        _write(args.output, {"error": f"load failed: {e}", "combined_score": 0.0})
        return

    if not hasattr(prog, "search_algorithm"):
        _write(args.output, {"error": "Missing search_algorithm function", "combined_score": 0.0})
        return

    search_algorithm = prog.search_algorithm

    costs = []
    transfer_times = []
    per_config = {}

    for cfg_name in CONFIG_FILENAMES:
        cfg_key = os.path.splitext(cfg_name)[0]
        cfg_path = os.path.join(CONFIGS_DIR, cfg_name)
        if not os.path.exists(cfg_path):
            continue

        config = json.loads(open(cfg_path, encoding="utf-8").read())
        src = config["source_node"]
        dsts = list(config["dest_nodes"])
        num_partitions = int(config["num_partitions"])

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(per_config_timeout)
        try:
            graph = copy.deepcopy(template_graph)
            t0 = time.time()
            topology = search_algorithm(src, dsts, graph, num_partitions)
            runtime = time.time() - t0

            # Validate topology before simulation (prevents partition-skipping hacks)
            validation_graph = copy.deepcopy(template_graph)
            is_valid, validation_error = validate_broadcast_topology(
                topology, src, dsts, num_partitions, validation_graph
            )
            if not is_valid:
                raise ValueError(f"Invalid topology: {validation_error}")

            simulator = BCSimulator(num_vms=num_vms, output_dir=None)
            transfer_time, cost = simulator.evaluate_path(topology, config)
            cost = float(cost)
            transfer_time = float(transfer_time)

            costs.append(cost)
            transfer_times.append(transfer_time)
            per_config[cfg_key] = {
                "cost": cost,
                "transfer_time": transfer_time,
                "runtime": runtime,
            }
            print(f"[OK] {cfg_key}: cost=${cost:.2f}  time={transfer_time:.1f}s  algo={runtime:.2f}s", flush=True)

        except _ConfigTimeout:
            costs.append(float("inf"))
            transfer_times.append(0.0)
            per_config[cfg_key] = {"error": f"timeout ({per_config_timeout}s)"}
            print(f"[TIMEOUT] {cfg_key}", flush=True)

        except Exception as e:
            costs.append(float("inf"))
            transfer_times.append(0.0)
            err_str = str(e)[:300]
            per_config[cfg_key] = {"error": err_str}
            print(f"[ERROR] {cfg_key}: {err_str}", flush=True)

        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    num_total = len(CONFIG_FILENAMES)
    num_ok = sum(1 for c in costs if math.isfinite(c))

    if num_ok == 0:
        _write(args.output, {
            "error": "All configs failed",
            "combined_score": 0.0,
            "num_configs": num_total,
            "num_succeeded": 0,
            "per_config": per_config,
        })
        return

    total_cost = sum(c for c in costs if math.isfinite(c))
    failed_penalty = (num_total - num_ok) * LOWER_COST
    total_cost += failed_penalty

    total_transfer_time = sum(transfer_times)

    cost_clamped = max(min(total_cost, LOWER_COST), UPPER_COST)
    normalized_score = (LOWER_COST - cost_clamped) / (LOWER_COST - UPPER_COST)

    result = {
        "combined_score": normalized_score,
        "total_cost": total_cost,
        "total_transfer_time": total_transfer_time,
        "num_configs": num_total,
        "num_succeeded": num_ok,
        "per_config": per_config,
    }
    _write(args.output, result)


if __name__ == "__main__":
    main()
