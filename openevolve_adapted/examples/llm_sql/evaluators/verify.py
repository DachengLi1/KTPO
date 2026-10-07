#!/usr/bin/env python3
"""
LLM-SQL Evaluation Worker — runs as subprocess for GIL-free parallelism.

Usage: python verify.py --program <path> --output <path> [--per-dataset-timeout 60]
"""
import sys
import os
import json
import time
import signal
import traceback
import importlib.util
import gc
import warnings
import itertools
import argparse

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::DeprecationWarning"

import pandas as pd
pd.options.mode.chained_assignment = None

EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(EVALUATOR_DIR)
sys.path.insert(0, TASK_DIR)
sys.path.insert(0, EVALUATOR_DIR)

from utils import evaluate_df_prefix_hit_cnt

DATASETS_DIR = os.path.join(TASK_DIR, "datasets")
DATASET_CONFIGS = [
    ("movies.csv", [["movieinfo", "movietitle", "rottentomatoeslink"]]),
    ("beer.csv", [["beer/beerId", "beer/name"]]),
    ("BIRD.csv", [["PostId", "Body"]]),
    ("PDMX.csv", [["path", "metadata"], ["hasmetadata", "isofficial", "isuserpublisher", "isdraft", "hasannotations", "subsetall"]]),
    ("products.csv", [["product_title", "parent_asin"]]),
]


class _DatasetTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _DatasetTimeout()


def _expected_ncols_after_merge(df: pd.DataFrame, col_merge) -> int:
    expected = df.shape[1]
    if not col_merge:
        return expected
    for group in col_merge:
        if not group:
            continue
        valid_cols = [c for c in group if c in df.columns]
        if len(valid_cols) >= 2:
            expected -= len(valid_cols) - 1
    return expected


def _build_expected_df(df: pd.DataFrame, col_merge, output_columns: list[str]):
    """Build expected DataFrame after col_merge for integrity checking."""
    expected_df = df.copy(deep=False)
    if not col_merge:
        return expected_df, None

    for group in col_merge:
        if not group:
            continue
        valid_cols = [c for c in group if c in expected_df.columns]
        if len(valid_cols) < 2:
            continue

        merge_name_to_order = {}
        for perm in itertools.permutations(valid_cols):
            merge_name_to_order["_".join(perm)] = list(perm)

        merge_candidates = [c for c in output_columns if c in merge_name_to_order]
        if len(merge_candidates) != 1:
            return None, f"expected exactly 1 merged column for {valid_cols}, found {len(merge_candidates)}"

        merged_col = merge_candidates[0]
        present_sources = [c for c in valid_cols if c in output_columns]
        if present_sources:
            return None, f"merged source columns still present: {present_sources}"

        merge_order = merge_name_to_order[merged_col]
        merged_series = expected_df[merge_order[0]].astype(str)
        for col in merge_order[1:]:
            merged_series = merged_series + expected_df[col].astype(str)
        expected_df = expected_df.drop(columns=valid_cols)
        if merged_col in expected_df.columns:
            return None, f"duplicate merged column name: {merged_col}"
        expected_df[merged_col] = merged_series

    return expected_df, None


def _validate_output_integrity(df: pd.DataFrame, col_merge, reordered: pd.DataFrame):
    """Row-level hash integrity check. Returns error string or None."""
    if reordered.columns.duplicated().any():
        return "duplicate column names"

    expected_df, err = _build_expected_df(df, col_merge, list(reordered.columns))
    if err:
        return err
    assert expected_df is not None

    if expected_df.columns.duplicated().any():
        return "duplicate column names after merge"

    if set(reordered.columns) != set(expected_df.columns):
        missing = sorted(set(expected_df.columns) - set(reordered.columns))
        extra = sorted(set(reordered.columns) - set(expected_df.columns))
        return f"columns mismatch: missing={missing} extra={extra}"

    def row_bag_hashes(frame):
        frame_str = frame.astype(str)
        sum_h = xor_h = sumsq_h = None
        for col in frame_str.columns:
            col_h = pd.util.hash_pandas_object(frame_str[col], index=False).to_numpy(dtype="uint64")
            if sum_h is None:
                sum_h, xor_h, sumsq_h = col_h.copy(), col_h.copy(), (col_h * col_h).copy()
            else:
                sum_h += col_h
                xor_h ^= col_h
                sumsq_h += col_h * col_h
        features = pd.DataFrame({"sum": sum_h, "xor": xor_h, "sumsq": sumsq_h})
        return pd.util.hash_pandas_object(features, index=False)

    expected_counts = row_bag_hashes(expected_df).value_counts().sort_index()
    output_counts = row_bag_hashes(reordered).value_counts().sort_index()
    if not expected_counts.equals(output_counts):
        return "rows/values differ from input after merges"
    return None


def _eval_one_dataset(program_module, csv_path, col_merge, timeout_sec):
    """Evaluate program on a single dataset. Returns (hit_rate, runtime) or raises."""
    import inspect
    accepts_parallel = "parallel" in inspect.signature(program_module.Solution.solve).parameters

    df = pd.read_csv(csv_path, low_memory=False)
    expected_rows = df.shape[0]
    expected_cols = _expected_ncols_after_merge(df, col_merge)

    solve_kwargs = dict(
        early_stop=100000, distinct_value_threshold=0.7,
        row_stop=4, col_stop=2, col_merge=col_merge,
    )
    if accepts_parallel:
        solve_kwargs["parallel"] = False

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_sec)
    try:
        start = time.time()
        reordered = program_module.Solution().solve(df.copy(deep=True), **solve_kwargs)
        runtime = time.time() - start
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if not isinstance(reordered, pd.DataFrame):
        raise ValueError(f"solve() returned {type(reordered).__name__}, expected DataFrame")
    if reordered.shape[0] != expected_rows or reordered.shape[1] != expected_cols:
        raise ValueError(f"shape ({expected_rows},{expected_cols}) expected, got {reordered.shape}")

    integrity_err = _validate_output_integrity(df, col_merge, reordered)
    if integrity_err:
        raise ValueError(integrity_err)

    _, hit_rate_pct = evaluate_df_prefix_hit_cnt(reordered)
    return hit_rate_pct / 100.0, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-dataset-timeout", type=int, default=60)
    args = parser.parse_args()

    per_ds_timeout = args.per_dataset_timeout

    # Load program
    try:
        spec = importlib.util.spec_from_file_location("program", args.program)
        prog = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prog)
    except Exception as e:
        _write(args.output, {"error": f"load failed: {e}", "combined_score": 0.0})
        return

    if not hasattr(prog, "Solution"):
        if hasattr(prog, "Evolved"):
            class _AutoSolution:
                def __init__(self):
                    self.alg = prog.Evolved()
                def solve(self, df, **kwargs):
                    reordered_df, _ = self.alg.reorder(df, **kwargs)
                    return reordered_df
            prog.Solution = _AutoSolution
        else:
            _write(args.output, {"error": "Missing Solution class and Evolved class", "combined_score": 0.0})
            return

    # Per-dataset evaluation
    hit_rates, runtimes, errors, per_ds = [], [], [], {}
    num_total = 0

    for ds_name, col_merge in DATASET_CONFIGS:
        csv_path = os.path.join(DATASETS_DIR, ds_name)
        if not os.path.exists(csv_path):
            continue
        num_total += 1
        ds_key = os.path.splitext(ds_name)[0].lower()

        try:
            hit_rate, runtime = _eval_one_dataset(prog, csv_path, col_merge, per_ds_timeout)
            hit_rates.append(hit_rate)
            runtimes.append(runtime)
            per_ds[ds_key] = {"hit_rate": hit_rate, "runtime": runtime}
            print(f"[OK] {ds_name}: hit={hit_rate*100:.2f}%  time={runtime:.1f}s", flush=True)

        except _DatasetTimeout:
            hit_rates.append(0.0)
            runtimes.append(float(per_ds_timeout))
            err_msg = f"timeout ({per_ds_timeout}s)"
            errors.append(f"{ds_name}: {err_msg}")
            per_ds[ds_key] = {"error": err_msg}
            print(f"[TIMEOUT] {ds_name}", flush=True)

        except Exception as e:
            hit_rates.append(0.0)
            runtimes.append(0.0)
            err_str = str(e)[:200]
            errors.append(f"{ds_name}: {err_str}")
            per_ds[ds_key] = {"error": err_str}
            print(f"[ERROR] {ds_name}: {err_str}", flush=True)

        finally:
            gc.collect()

    if num_total == 0:
        _write(args.output, {"error": "No datasets found", "combined_score": 0.0})
        return

    avg_hit = sum(hit_rates) / num_total
    avg_rt = sum(runtimes) / num_total
    combined = 0.95 * avg_hit + 0.05 * (12 - min(12, avg_rt)) / 12

    result = {
        "combined_score": combined,
        "average_hit_rate": avg_hit,
        "average_runtime": avg_rt,
        "total_runtime": sum(runtimes),
        "num_datasets": num_total,
        "num_succeeded": sum(1 for h in hit_rates if h > 0),
        "per_dataset": per_ds,
        "hit_rates": hit_rates,
        "runtimes": runtimes,
    }
    if errors:
        result["errors"] = errors

    _write(args.output, result)


def _write(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
