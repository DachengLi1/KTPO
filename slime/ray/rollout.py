import logging
import multiprocessing
import random
import threading
import time
from pathlib import Path
from typing import List, Union

import ray
import torch
import wandb
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.ray.rollout_data_source import RolloutDataSourceWithBuffer
from slime.utils.http_utils import find_available_port, get_host_info, init_http_client
from slime.utils.misc import load_function
from slime.utils.ray_utils import Box
from slime.utils.types import Sample
from slime.utils.wandb_utils import init_wandb_secondary

from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg, wandb_run_id):
        self.args = args
        self.pg = pg
        self._openai_mode = getattr(args, "openai_api_base", None) is not None

        if not self._openai_mode:
            _start_router(args)
        init_wandb_secondary(args, wandb_run_id)
        if not self._openai_mode:
            init_http_client(args)

        self.data_source = RolloutDataSourceWithBuffer(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        print(f"import {self.args.rollout_function_path} as generate_rollout function.")
        print(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only or self._openai_mode:
            # No sglang engines needed
            self.all_rollout_engines = []
        else:
            num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
            num_engines = args.rollout_num_gpus // num_gpu_per_engine
            self.all_rollout_engines = [None] * num_engines
        self.num_new_engines = init_rollout_engines(args, pg, self.all_rollout_engines)
        self.nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // max(1, args.num_gpus_per_node))
        # when doing multi-node serving, we will only send request to node-0 for each engine.
        self.rollout_engines = self.all_rollout_engines[:: max(1, self.nodes_per_engine)]
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()

        # fault tolerance
        self._health_monitor_thread = None
        self._health_monitor_stop_event = None
        self._health_check_interval = args.rollout_health_check_interval
        self._health_check_timeout = args.rollout_health_check_timeout
        self._health_check_first_wait = args.rollout_health_check_first_wait

    def get_rollout_engines_and_lock(self):
        return self.rollout_engines, self.rollout_engine_lock, self.num_new_engines

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def get_cum_actual_evals(self) -> int:
        """Return the cumulative number of programs evaluated so far.

        This is maintained by _log_rollout_data as a function-level attribute
        and gives an honest count of LLM calls made (respects adaptive-k early
        termination).  Used by train.py to enforce --sample-budget.
        """
        return int(getattr(_log_rollout_data, "_cum_actual_evals", 0))

    def generate(self, rollout_id):
        self.rollout_id = rollout_id
        monitor_started = self._start_health_monitor()
        start_time = time.time()
        try:
            data = self._get_rollout_data()
            self._save_debug_rollout_data(data)
            _log_rollout_data(rollout_id, self.args, data, time.time() - start_time)
            data = self._convert_samples_to_train_data(data)
            return Box(ray.put(data))
        finally:
            if monitor_started:
                self._stop_health_monitor()
                # Remove unpicklable attributes from args before passing to Ray actors
                # (sglang_rollout.py stores _acds_gym on args which contains threading.Lock)
                _acds_gym_backup = getattr(self.args, "_acds_gym", None)
                if hasattr(self.args, "_acds_gym"):
                    delattr(self.args, "_acds_gym")
                self.num_new_engines = init_rollout_engines(self.args, self.pg, self.all_rollout_engines)
                self.rollout_engines = self.all_rollout_engines[:: self.nodes_per_engine]
                if _acds_gym_backup is not None:
                    self.args._acds_gym = _acds_gym_backup

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        # TODO: add fault tolerance to eval
        data = self.eval_generate_rollout(self.args, rollout_id, self.data_source, evaluation=True)
        _log_eval_rollout_data(rollout_id, self.args, data)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        result = self.data_source.load(rollout_id)
        # Log initial program score at inner_step=0 so step-0 appears in W&B
        _log_initial_best_score(self.args, self.data_source)
        return result

    def offload(self):
        """Release memory occupation across all engines."""
        if not self.args.offload_rollout:
            return
        handles = [engine.release_memory_occupation.remote() for engine in self.rollout_engines if engine is not None]
        if handles:
            ray.get(handles)

    def onload_weights(self):
        """Restore weights on all engines (resume from CPU cache)."""
        if not self.args.offload_rollout:
            return
        handles = [
            engine.resume_memory_occupation.remote(tags=["weights"])
            for engine in self.rollout_engines if engine is not None
        ]
        if handles:
            ray.get(handles)

    def onload_kv(self):
        """Resume KV cache and CUDA graphs on all engines."""
        if not self.args.offload_rollout:
            return
        handles = [
            engine.resume_memory_occupation.remote(tags=["kv_cache", "cuda_graph"])
            for engine in self.rollout_engines if engine is not None
        ]
        if handles:
            ray.get(handles)

    def _start_health_monitor(self) -> bool:
        if not self.rollout_engines:
            return False

        # Skip health monitor for single-engine setups — health checks can
        # timeout during long generations and kill the only engine.
        if len(self.rollout_engines) <= 1:
            return False

        assert self._health_monitor_thread is None, "Health monitor thread is already running."

        self._health_monitor_stop_event = threading.Event()
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        self._health_monitor_thread.start()
        return True

    def _stop_health_monitor(self) -> None:
        if not self._health_monitor_thread:
            return

        assert self._health_monitor_stop_event is not None
        self._health_monitor_stop_event.set()
        timeout = self._health_check_timeout + self._health_check_interval + 5
        self._health_monitor_thread.join(timeout=timeout)
        if self._health_monitor_thread.is_alive():
            logging.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)

        self._health_monitor_thread = None
        self._health_monitor_stop_event = None

    def _health_monitor_loop(self) -> None:
        assert self._health_monitor_stop_event is not None
        # TODO: need to be waiting for the large moe to be ready. this is hacky.
        if self._health_monitor_stop_event.wait(self._health_check_first_wait):
            return
        while not self._health_monitor_stop_event.is_set():
            self._run_health_checks()
            if self._health_monitor_stop_event.wait(self._health_check_interval):
                break

    def _run_health_checks(self) -> None:
        for rollout_engine_id, engine in enumerate(self.rollout_engines):
            if self._health_monitor_stop_event is not None and self._health_monitor_stop_event.is_set():
                break
            self._check_engine_health(rollout_engine_id, engine)

    def _check_engine_health(self, rollout_engine_id, engine) -> None:
        if engine is None:
            return

        try:
            ray.get(engine.health_generate.remote(timeout=self._health_check_timeout))
        except Exception as e:
            print(f"Health check timed out for rollout engine {rollout_engine_id} (ray timeout). Killing actor.")
            for i in range(rollout_engine_id * self.nodes_per_engine, (rollout_engine_id + 1) * self.nodes_per_engine):
                engine = self.all_rollout_engines[i]
                try:
                    ray.kill(engine)
                except Exception:
                    pass
                self.all_rollout_engines[i] = None
            self.rollout_engines[rollout_engine_id] = None

    def _get_rollout_data(self):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                open(self.args.load_debug_rollout_data.format(rollout_id=self.rollout_id), "rb"),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
        else:
            data = self.generate_rollout(self.args, self.rollout_id, self.data_source, evaluation=False)
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = sum(data, [])

            if len(data) % self.args.global_batch_size != 0:
                trim_len = (len(data) // self.args.global_batch_size) * self.args.global_batch_size
                origin_data_length = len(data)
                data = data[:trim_len]
                print(f"trim number of samples from {origin_data_length} to {trim_len}")
        return data

    def _save_debug_rollout_data(self, data):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=self.rollout_id))
            print(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(
                    rollout_id=self.rollout_id,
                    samples=[sample.to_dict() for sample in data],
                ),
                path,
            )

    # ── Reward transforms (applied before GRPO group normalization) ─────

    # ── Main reward post-processing ──────────────────────────────────────

    def _post_process_rewards(self, samples: Union[list[Sample], list[list[Sample]]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]

        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] % self.args.n_samples_per_prompt == 0:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])

            # ── GRPO group normalization ──────────────────────────────
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean
            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)
            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: Union[list[Sample], list[list[Sample]]]):
        """
        Convert inference generated samples to training data.
        """
        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length
            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        return train_data


def init_rollout_engines(args, pg, all_rollout_engines):
    if args.debug_train_only or getattr(args, "openai_api_base", None):
        return 0

    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
    num_engines = args.rollout_num_gpus // num_gpu_per_engine
    assert len(all_rollout_engines) == num_engines

    pg, reordered_bundle_indices = pg

    RolloutRayActor = ray.remote(SGLangEngine)

    rollout_engines = []
    for i in range(num_engines):
        if all_rollout_engines[i] is not None:
            continue

        num_gpus = 0.2
        num_cpus = num_gpus

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        rollout_engine = RolloutRayActor.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={
                "env_vars": {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}
                | {
                    "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                }
            },
        ).remote(args, rank=i)

        rollout_engines.append((i, rollout_engine))
        all_rollout_engines[i] = rollout_engine

    num_new_engines = len(rollout_engines)

    if num_new_engines == 0:
        return num_new_engines

    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    num_engines_per_node = max(
        1, min(args.num_gpus_per_node, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine
    )
    addr_and_ports = [{} for _ in range(num_engines)]

    visited_nodes = set()
    for rank, engine in rollout_engines:
        if rank // num_engines_per_node in visited_nodes:
            continue
        visited_nodes.add(rank // num_engines_per_node)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        def get_addr_and_ports():
            # use small ports to prevent ephemeral port between 32768 and 65536.
            start_port = 10000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports()

        for i in range(num_engines_on_this_node):
            addr_and_ports[rank + i]["port"] = get_port()
            addr_and_ports[rank + i]["nccl_port"] = get_port()

        if args.rollout_num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = args.rollout_num_gpus_per_engine // args.num_gpus_per_node
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        print(f"Ports for engine {i}: {addr_and_ports[i]}")

    # TODO: don't ray.get here to overlap train actor init with rollout engine init.
    # somehow if we don't sync here, the --debug-rollout-only mode will crash.
    init_handles = [engine.init.remote(**(addr_and_ports[rank])) for rank, engine in rollout_engines]
    ray.get(init_handles)
    return num_new_engines


def _start_router(args):
    """start sgl router and slime router"""
    if args.sglang_router_ip is not None:
        return

    args.sglang_router_ip = get_host_info()[1]
    args.sglang_router_port = find_available_port(random.randint(3000, 4000))

    if args.use_slime_router:
        from slime.router.router import run_router

        router_args = args

    else:
        from sglang_router.launch_router import RouterArgs

        from slime.utils.http_utils import run_router

        args.sglang_router_ip = get_host_info()[1]
        args.sglang_router_port = find_available_port(random.randint(3000, 4000))

        router_args = RouterArgs(
            host=args.sglang_router_ip,
            port=args.sglang_router_port,
            balance_abs_threshold=0,
            prometheus_port=find_available_port(random.randint(4000, 5000)),
        )

        if hasattr(router_args, "log_level"):
            router_args.log_level = "warn"

        if hasattr(router_args, "request_timeout_secs"):
            router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    print(f"Router launched at {args.sglang_router_ip}:{args.sglang_router_port}")


def _log_eval_rollout_data(rollout_id, args, data):
    log_dict = {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)

    print(f"eval {rollout_id}: {log_dict}")
    if args.use_wandb:
        log_dict["eval/step"] = (
            rollout_id
            if not args.wandb_always_use_train_step
            else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
        )
        wandb.log(log_dict)

    if args.use_tensorboard:
        from slime.utils.tensorboard_utils import _TensorboardAdapter

        tb = _TensorboardAdapter(args)
        tb.log(
            data=log_dict,
            step=(
                rollout_id
                if not args.wandb_always_use_train_step
                else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
            ),
        )


def _log_rollout_data(rollout_id, args, samples, rollout_time):
    if args.load_debug_rollout_data:
        return

    log_dict = {}
    response_lengths = [
        sum(sample.loss_mask) if sample.loss_mask is not None else sample.response_length for sample in samples
    ]
    log_dict["perf/rollout_time"] = rollout_time
    if args.rollout_num_gpus:
        log_dict["perf/tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
    log_dict["perf/longest_sample_tokens_per_sec"] = max(response_lengths) / rollout_time

    # EvolvingGym diff-format compliance diagnostics.
    try:
        no_diff = 0
        has_err = 0
        for s in samples:
            r = getattr(s, "reward", None)
            if isinstance(r, dict):
                artifacts = r.get("artifacts", None)
                err_t = None
                if isinstance(artifacts, dict):
                    err_t = artifacts.get("error_type", None)
                if err_t is not None:
                    has_err += 1
                if err_t == "no_diff_blocks":
                    no_diff += 1
        n = len(samples) if samples is not None else 0
        if n > 0:
            log_dict["rollout/no_diff_blocks_frac"] = float(no_diff) / float(n)
            log_dict["rollout/no_diff_blocks_count"] = float(no_diff)
            log_dict["rollout/error_type_annotated_frac"] = float(has_err) / float(n)
    except Exception:
        pass

    # ---------------------------------------------------------------------------
    # Multi-step refinement diagnostics
    # ---------------------------------------------------------------------------
    try:
        from slime.rollout.sglang_rollout import get_effective_k
        multistep_k = get_effective_k(args, rollout_id)
        log_dict["rollout/multistep/effective_k"] = multistep_k
        if multistep_k > 0 and samples:
            import statistics
            all_scores: list[list[float]] = []
            all_rewards: list[float] = []
            reward_key = getattr(args, "reward_key", "reward")
            n = len(samples)

            for s in samples:
                r = getattr(s, "reward", None)
                if isinstance(r, dict):
                    ms = r.get("multistep_raw_scores")
                    if ms and isinstance(ms, (list, tuple)):
                        all_scores.append([float(x) for x in ms])
                    rv = r.get(reward_key, 0.0)
                    all_rewards.append(float(rv) if isinstance(rv, (int, float)) else 0.0)
                else:
                    all_rewards.append(float(r) if isinstance(r, (int, float)) else 0.0)

            # --- Accumulated reward distribution ---
            if all_rewards:
                log_dict["rollout/multistep/reward_mean"] = statistics.mean(all_rewards)
                log_dict["rollout/multistep/reward_std"] = statistics.pstdev(all_rewards)
                log_dict["rollout/multistep/reward_min"] = min(all_rewards)
                log_dict["rollout/multistep/reward_max"] = max(all_rewards)
                log_dict["rollout/multistep/reward_positive_frac"] = sum(1 for x in all_rewards if x > 0) / len(all_rewards)

            # --- Per-step score analysis ---
            if all_scores:
                num_steps = max(len(sc) for sc in all_scores)
                for step_i in range(num_steps):
                    vals = [sc[step_i] for sc in all_scores if step_i < len(sc)]
                    if vals:
                        label = "parent" if step_i == 0 else str(step_i)
                        log_dict[f"rollout/multistep/avg_score_step_{label}"] = statistics.mean(vals)

                # Best-step improvement (diagnostic only)
                best_improvements = []
                final_improvements = []
                for sc in all_scores:
                    if len(sc) >= 2:
                        best_improvements.append(max(sc[1:]) - sc[0])
                        final_improvements.append(sc[-1] - sc[0])
                if best_improvements:
                    log_dict["rollout/multistep/avg_best_improvement"] = statistics.mean(best_improvements)
                    log_dict["rollout/multistep/best_improvement_rate"] = sum(1 for x in best_improvements if x > 1e-6) / len(best_improvements)
                    log_dict["rollout/multistep/avg_final_improvement"] = statistics.mean(final_improvements)
                    log_dict["rollout/multistep/final_improvement_rate"] = sum(1 for x in final_improvements if x > 1e-6) / len(final_improvements)

                # Per-step diagnostics using explicit validity flags
                total_invalid = 0
                total_lazy = 0
                total_pos_delta = 0
                total_steps = 0
                all_validities: list[list[bool]] = []
                for s in samples:
                    r = getattr(s, "reward", None)
                    if isinstance(r, dict):
                        mv = r.get("multistep_validities")
                        if mv and isinstance(mv, (list, tuple)):
                            all_validities.append(list(mv))
                for sc, sv in zip(all_scores, all_validities if all_validities else [[True]*len(sc) for sc in all_scores]):
                    for t in range(1, len(sc)):
                        total_steps += 1
                        delta = sc[t] - sc[t - 1]
                        step_valid = sv[t] if t < len(sv) else (sc[t] > 0)
                        if not step_valid:
                            total_invalid += 1
                        if abs(delta) < 1e-6:
                            total_lazy += 1
                        if delta > 1e-6:
                            total_pos_delta += 1

                ns = len(all_scores)
                log_dict["rollout/multistep/avg_invalid_steps"] = total_invalid / ns
                log_dict["rollout/multistep/avg_lazy_steps"] = total_lazy / ns
                log_dict["rollout/multistep/avg_positive_delta_steps"] = total_pos_delta / ns
                if total_steps > 0:
                    log_dict["rollout/multistep/invalid_step_frac"] = total_invalid / total_steps
                    log_dict["rollout/multistep/lazy_step_frac"] = total_lazy / total_steps
                    log_dict["rollout/multistep/positive_delta_step_frac"] = total_pos_delta / total_steps

            # --- Token statistics ---
            total_response_toks = 0
            total_llm_toks = 0
            for s in samples:
                resp_len = int(getattr(s, "response_length", 0) or 0)
                total_response_toks += resp_len
                lm = getattr(s, "loss_mask", None)
                if lm is not None:
                    total_llm_toks += sum(int(x) for x in lm)
                else:
                    total_llm_toks += resp_len
            if n > 0:
                log_dict["rollout/multistep/avg_total_tokens"] = total_response_toks / n
                log_dict["rollout/multistep/avg_llm_tokens"] = total_llm_toks / n
                if total_response_toks > 0:
                    log_dict["rollout/multistep/feedback_token_ratio"] = 1.0 - (total_llm_toks / total_response_toks)

            # --- Per-step response length statistics ---
            all_resp_lens: list[list[int]] = []
            max_resp_len_limit = int(getattr(args, "rollout_max_response_len", 0))
            n_truncated = 0
            for s in samples:
                r = getattr(s, "reward", None)
                if isinstance(r, dict):
                    rl = r.get("multistep_resp_lens")
                    if rl and isinstance(rl, (list, tuple)):
                        all_resp_lens.append([int(x) for x in rl])
                        # Check if the last step hit the limit (truncated)
                        if max_resp_len_limit > 0 and int(rl[-1]) >= max_resp_len_limit:
                            n_truncated += 1

            if all_resp_lens:
                num_steps = max(len(rl) for rl in all_resp_lens)
                step_len_strs = []
                for step_i in range(num_steps):
                    vals = [rl[step_i] for rl in all_resp_lens if step_i < len(rl)]
                    if vals:
                        avg_val = statistics.mean(vals)
                        max_val = max(vals)
                        log_dict[f"rollout/multistep/avg_resp_len_step_{step_i+1}"] = avg_val
                        log_dict[f"rollout/multistep/max_resp_len_step_{step_i+1}"] = max_val
                        step_len_strs.append(f"s{step_i+1}={avg_val:.0f}(max={max_val})")
                # Final-step stats
                final_lens = [rl[-1] for rl in all_resp_lens]
                log_dict["rollout/multistep/avg_final_resp_len"] = statistics.mean(final_lens)
                log_dict["rollout/multistep/max_final_resp_len"] = max(final_lens)
                log_dict["rollout/multistep/truncated_frac"] = n_truncated / len(all_resp_lens)
                resp_len_summary = " ".join(step_len_strs)
            else:
                resp_len_summary = "N/A"
                
            print(f"[MultiStep logging] rollout_id={rollout_id} "
                  f"reward_mean={log_dict.get('rollout/multistep/reward_mean', 'N/A')} "
                  f"best_impr_rate={log_dict.get('rollout/multistep/best_improvement_rate', 'N/A')} "
                  f"final_impr_rate={log_dict.get('rollout/multistep/final_improvement_rate', 'N/A')} "
                  f"invalid_frac={log_dict.get('rollout/multistep/invalid_step_frac', 'N/A')} "
                  f"lazy_frac={log_dict.get('rollout/multistep/lazy_step_frac', 'N/A')} "
                  f"avg_llm_tokens={log_dict.get('rollout/multistep/avg_llm_tokens', 'N/A')} "
                  f"resp_lens=[{resp_len_summary}] "
                  f"truncated={log_dict.get('rollout/multistep/truncated_frac', 'N/A')}")
    except Exception as e:
        print(f"[MultiStep logging] Failed at rollout_id={rollout_id}: {e}")

    try:
        from slime.rollout.sglang_rollout import get_inner_step_cum, get_effective_k
        inner_step_cum = int(get_inner_step_cum(args, int(rollout_id)))
    except Exception:
        inner_step_cum = int(rollout_id) + 1

    log_dict["rollout/multistep/inner_step_cum"] = inner_step_cum

    # ── cum_actual_evals: counts programs actually evaluated (fair across k and adaptive-k) ──
    # = Σ_{r=0}^{rollout_id} (actual steps taken across all chains in rollout r)
    # For non-adaptive-k runs: actual_steps == configured_k, so this equals
    # inner_step_cum × n_samples (fully backward-compatible).
    # For adaptive-k runs: early-terminated chains contribute fewer than k, giving
    # an honest count of LLM calls actually made.
    try:
        n_samples = int(getattr(args, "n_samples_per_prompt", 1))
        _configured_k = int(get_effective_k(args, int(rollout_id)))
        # Read actual steps for this rollout from sample reward dicts
        _actual_steps_this = []
        for _s in (samples or []):
            _r = getattr(_s, "reward", None)
            if isinstance(_r, dict):
                _ak = _r.get("chain_actual_k", None)
                if isinstance(_ak, int) and _ak > 0:
                    _actual_steps_this.append(_ak)
        if not _actual_steps_this:
            # Fallback: assume full k (old rollouts without chain_actual_k)
            _actual_steps_this = [_configured_k] * max(1, len(samples or []))
        _evals_this_rollout = sum(_actual_steps_this)  # total steps across all n_samples chains
        # Accumulate across rollouts using a function-level attribute
        if not hasattr(_log_rollout_data, "_cum_actual_evals"):
            _log_rollout_data._cum_actual_evals = 0
        _log_rollout_data._cum_actual_evals += _evals_this_rollout
        _cum_actual_evals = _log_rollout_data._cum_actual_evals
        _avg_actual_k = sum(_actual_steps_this) / max(1, len(_actual_steps_this))
        log_dict["rollout/multistep/avg_actual_k"] = _avg_actual_k
        log_dict["rollout/multistep/cum_actual_evals"] = _cum_actual_evals

        # ── Adaptive-k diagnostics ──
        # Log how many chains terminated early vs ran the full k steps.
        _adaptive_patience = int(getattr(args, "adaptive_k_patience", 0))
        log_dict["rollout/multistep/adaptive_k_patience"] = _adaptive_patience
        if _adaptive_patience > 0 and _configured_k > 0:
            # A chain is "early-stopped" iff it ran fewer than configured k steps
            _n_early_stopped = sum(1 for ak in _actual_steps_this if ak < _configured_k)
            _n_chains = max(1, len(_actual_steps_this))
            _early_stop_frac = _n_early_stopped / _n_chains
            _steps_saved = sum(max(0, _configured_k - ak) for ak in _actual_steps_this)
            _steps_possible = _n_chains * _configured_k
            _saved_frac = _steps_saved / _steps_possible if _steps_possible > 0 else 0.0
            log_dict["rollout/multistep/adaptive_k_early_stop_count"] = _n_early_stopped
            log_dict["rollout/multistep/adaptive_k_early_stop_frac"] = _early_stop_frac
            log_dict["rollout/multistep/adaptive_k_steps_saved"] = _steps_saved
            log_dict["rollout/multistep/adaptive_k_saved_frac"] = _saved_frac
            print(
                f"[AdaptiveK logging] rollout={rollout_id}  patience={_adaptive_patience}  "
                f"early_stopped={_n_early_stopped}/{_n_chains} ({_early_stop_frac:.1%})  "
                f"steps_saved={_steps_saved}/{_steps_possible} ({_saved_frac:.1%})  "
                f"avg_actual_k={_avg_actual_k:.2f}/{_configured_k}",
                flush=True,
            )

        print(
            f"[CumEvals] rollout={rollout_id}  evals_this={_evals_this_rollout} "
            f"(avg_actual_k={_avg_actual_k:.1f}/{_configured_k}×{n_samples}chains)  "
            f"cum_total={_cum_actual_evals}",
            flush=True,
        )
    except Exception as _e:
        print(f"[CumEvals] Failed at rollout_id={rollout_id}: {_e}")
        _cum_actual_evals = inner_step_cum * int(getattr(args, "n_samples_per_prompt", 1))
    print(f"perf {rollout_id}: {log_dict}")
    if args.use_wandb:
        log_dict["rollout/step"] = int(rollout_id)

        # --- Combined per-step score line chart (all steps on one plot) ---
        try:
            step_keys = sorted(
                [k for k in log_dict if k.startswith("rollout/multistep/avg_score_step_")],
                key=lambda k: -1 if k.endswith("_parent") else int(k.rsplit("_", 1)[-1]),
            )
            if step_keys:
                # Accumulate history in a function attribute
                if not hasattr(_log_rollout_data, "_score_history"):
                    _log_rollout_data._score_history = {}  # name -> ([xs], [ys])
                hist = _log_rollout_data._score_history
                _x_val = log_dict.get("rollout/multistep/cum_actual_evals", inner_step_cum)
                for k in step_keys:
                    label = k.rsplit("_", 1)[-1]
                    name = "s0_parent" if label == "parent" else f"s{label}"
                    if name not in hist:
                        hist[name] = ([], [])
                    hist[name][0].append(_x_val)
                    hist[name][1].append(log_dict[k])
                # Build line_series with all accumulated data
                series_names = sorted(
                    hist.keys(),
                    key=lambda n: -1 if "parent" in n else int(n[1:]),
                )
                xs = [hist[n][0] for n in series_names]
                ys = [hist[n][1] for n in series_names]
                log_dict["rollout/multistep/score_all_steps"] = wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=series_names,
                    title="Avg Score by Refinement Step",
                    xname="programs_evaluated",
                )
        except Exception:
            pass  # Don't let chart creation break logging

        # In shared mode, explicit step is ignored; rely on define_metric step bindings.
        wandb.log(log_dict)

    if args.use_tensorboard:
        from slime.utils.tensorboard_utils import _TensorboardAdapter

        tb = _TensorboardAdapter(args)
        tb.log(
            data=log_dict,
            step=(
                rollout_id
                if not args.wandb_always_use_train_step
                else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
            ),
        )


def _log_initial_best_score(args, data_source) -> None:
    """Log the initial (seed) program's best_score at inner_step=0 so step-0 appears in W&B."""
    if not getattr(args, "use_wandb", False):
        return
    if not getattr(args, "evolving_gym", False):
        return
    try:
        gym = getattr(getattr(data_source, "evolving_gym_manager", None), "gym", None)
        if gym is None:
            return
        best_program = gym.database.get_best_program()
        if best_program is None or not best_program.metrics:
            return
        best_score = best_program.metrics.get("combined_score", None)
        if not isinstance(best_score, (int, float)):
            return
        wandb_dict = {
            "evolving_gym/best_score": float(best_score),
            "evolving_gym/inner_step": 0,
            "evolving_gym/database_size": float(len(getattr(gym.database, "programs", {}) or {})),
            "evolving_gym/step": -1,
        }
        wandb.log(wandb_dict)
        print(f"[W&B] Logged initial program score at step=0: best_score={best_score}")
    except Exception as e:
        print(f"[W&B] Failed to log initial best score: {e}")


def _log_best_objective(rollout_id, args, data_source) -> None:
    """Log the current best objective (combined_score) to wandb for evolving_gym runs."""
    if not getattr(args, "use_wandb", False):
        return
    if not getattr(args, "evolving_gym", False):
        return
    try:
        from slime.rollout.sglang_rollout import get_inner_step_cum
        gym = getattr(getattr(data_source, "evolving_gym_manager", None), "gym", None)
        if gym is None:
            return
        best_prog = gym.database.get_best_program(metric="combined_score")
        if best_prog is None:
            return
        best_score = best_prog.metrics.get("combined_score", None)
        if not isinstance(best_score, (int, float)):
            return

        # Prefer cum_actual_evals (honest count of programs evaluated) as x-axis.
        # Fall back to inner_step_cum × n_samples for backward compatibility.
        _cum_actual = getattr(_log_rollout_data, "_cum_actual_evals", None)
        if _cum_actual is None:
            _n_samples = int(getattr(args, "n_samples_per_prompt", 1))
            _cum_actual = int(get_inner_step_cum(args, int(rollout_id))) * _n_samples

        wandb.log(
            {
                "objective/best_combined_score": float(best_score),
                "objective/database_size": float(len(getattr(gym.database, "programs", {}) or {})),
                "objective/rollout_id": int(rollout_id),
                "objective/inner_step": int(get_inner_step_cum(args, int(rollout_id))),  # kept for compat
                "objective/cum_actual_evals": int(_cum_actual),
            }
        )
    except Exception as e:
        print(f"[W&B] Failed to log best objective at rollout_id={rollout_id}: {e}")
