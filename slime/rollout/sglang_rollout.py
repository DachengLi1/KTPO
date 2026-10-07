import asyncio
import base64
import copy
import io
import json
import math
import os
import re
import random
import time
import uuid
from argparse import Namespace
from typing import Any, Callable, Optional, Union

import wandb
from PIL import Image
import sglang_router
from packaging.version import parse
from tqdm import tqdm
from transformers import AutoTokenizer

from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm
from .rm_hub.evolving_gym_rm import get_database_worst_score

__all__ = ["generate_rollout"]

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NOTE: Prompt instructions & feedback are now task-generic              ║
# ║  (see evolve_prompt_utils.py).  Remaining hardcodes:                    ║
# ║                                                                          ║
# ║  1. _classify_error_code — magic error-code thresholds (-0.1 … -0.4)    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ---------------------------------------------------------------------------
# Curriculum K schedule helper
# ---------------------------------------------------------------------------
def _parse_k_schedule(schedule_str: str | None) -> list[tuple[int, int]] | None:
    """Parse '--multistep-k-schedule' value into [(k, n_rollouts), ...].

    Example: '1:31,4:23'  →  [(1, 31), (4, 23)]
    Returns None if schedule_str is None or empty.
    """
    if not schedule_str:
        return None
    segments = []
    for part in schedule_str.split(","):
        k_str, n_str = part.strip().split(":")
        segments.append((int(k_str), int(n_str)))
    return segments


def get_effective_k(args: Namespace, rollout_id: int) -> int:
    """Return the multistep K that should be used for *rollout_id*.

    If --multistep-k-schedule is set, the schedule takes priority.
    Otherwise falls back to --multistep-refine-k.
    """
    schedule = _parse_k_schedule(getattr(args, "multistep_k_schedule", None))
    if schedule is None:
        return int(getattr(args, "multistep_refine_k", 0))

    cumulative = 0
    for seg_idx, (k, n) in enumerate(schedule):
        cumulative += n
        if rollout_id < cumulative:
            return k
    # Past the schedule – use the last segment's K
    return schedule[-1][0]



def get_rollout_x_advance(args: Namespace, rollout_id: int) -> int:
    """How much x-axis should advance for this rollout."""
    return max(1, int(get_effective_k(args, rollout_id)))


# Cumulative actual evals counter — updated after each rollout in _evolving_gym_log.
# Used as the wandb x-axis so that adaptive-k and flat-k are comparable on the same scale.
_cum_actual_evals_wandb: int = 0

# ── Inner-step periodic wandb logger ──
# Logs lightweight metrics every _INNER_LOG_EVERY evaluations so the user
# can monitor progress during long multi-step rollouts without waiting for
# the full rollout to complete.
_inner_eval_count: int = 0
_inner_valid_count: int = 0
_inner_score_sum: float = 0.0
_inner_best_chain_score: float = float("-inf")
_INNER_LOG_EVERY: int = int(os.environ.get("LEARNX_INNER_LOG_EVERY", "48"))


def _inner_step_wandb_log(
    args,
    step_idx: int,
    k: int,
    score: float,
    is_valid: bool,
):
    """Lightweight wandb log called after each inner-step RM evaluation.

    Logs every _INNER_LOG_EVERY evaluations with:
      - Current DB best score (real-time)
      - Running valid rate and mean valid score since last log
      - Total inner eval count
    """
    global _inner_eval_count, _inner_valid_count, _inner_score_sum, _inner_best_chain_score

    _inner_eval_count += 1
    if is_valid and score > 0:
        _inner_valid_count += 1
        _inner_score_sum += score
    if score > _inner_best_chain_score:
        _inner_best_chain_score = score

    if _inner_eval_count % _INNER_LOG_EVERY != 0:
        return
    if not (getattr(args, "use_wandb", False) and wandb.run):
        return

    try:
        from .rm_hub.evolving_gym_rm import get_database_best_score, get_database_size
        db_best = get_database_best_score()
        db_size = get_database_size()

        valid_rate = _inner_valid_count / _INNER_LOG_EVERY if _INNER_LOG_EVERY > 0 else 0
        mean_valid = _inner_score_sum / _inner_valid_count if _inner_valid_count > 0 else 0

        log_dict = {
            "inner/eval_count": _inner_eval_count,
            "inner/valid_rate": valid_rate,
            "inner/mean_valid_score": mean_valid,
            "inner/best_chain_score": _inner_best_chain_score,
            "inner/db_size": db_size,
        }
        if db_best is not None:
            log_dict["inner/db_best_score"] = db_best

        wandb.log(log_dict, commit=True)  # commit so data appears in wandb immediately
        print(
            f"[InnerLog] evals={_inner_eval_count} "
            f"db_best={db_best} db_size={db_size} "
            f"valid_rate={valid_rate:.2f} mean_valid={mean_valid:.4f} "
            f"best_chain={_inner_best_chain_score:.4f}",
            flush=True,
        )

        # Reset running counters for next interval
        _inner_valid_count = 0
        _inner_score_sum = 0.0
    except Exception as e:
        print(f"[InnerLog] wandb log failed: {e}", flush=True)


def get_inner_step_cum(args: Namespace, rollout_id: int) -> int:
    """Cumulative inner-step index up to and including rollout_id."""
    if rollout_id < 0:
        return 0
    total = 0
    for rid in range(int(rollout_id) + 1):
        total += get_rollout_x_advance(args, rid)
    return total


def _log_schedule_info(args: Namespace, rollout_id: int) -> None:
    """Print a one-line summary of where *rollout_id* sits in the K schedule."""
    schedule = _parse_k_schedule(getattr(args, "multistep_k_schedule", None))
    if schedule is None:
        k = int(getattr(args, "multistep_refine_k", 0))
        print(f"[K-schedule] rollout={rollout_id}  flat K={k}", flush=True)
        return

    effective_k = get_effective_k(args, rollout_id)
    cumulative = 0
    phase_str_parts = []
    current_phase = -1
    for seg_idx, (k, n) in enumerate(schedule):
        start = cumulative
        end = cumulative + n
        marker = " ◀" if start <= rollout_id < end else ""
        phase_str_parts.append(f"K={k}×{n}[{start}-{end - 1}]{marker}")
        if start <= rollout_id < end:
            current_phase = seg_idx
            phase_progress = rollout_id - start
            phase_total = n
        cumulative += n

    schedule_summary = " | ".join(phase_str_parts)
    if current_phase >= 0:
        print(
            f"[K-schedule] rollout={rollout_id}  K={effective_k}  "
            f"phase {current_phase + 1}/{len(schedule)} ({phase_progress + 1}/{phase_total})  "
            f"schedule: {schedule_summary}",
            flush=True,
        )
    else:
        # Past the schedule
        print(
            f"[K-schedule] rollout={rollout_id}  K={effective_k}  "
            f"(past schedule, using last K)  schedule: {schedule_summary}",
            flush=True,
        )


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistant state for the generation process
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

        _openai_mode = getattr(args, "openai_api_base", None) is not None
        if _openai_mode:
            # OpenAI backend: concurrency controlled by openai_generate.py's semaphore
            self.semaphore = asyncio.Semaphore(getattr(args, "openai_concurrency", 64))
        else:
            self.semaphore = asyncio.Semaphore(
                args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        self.reset()

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False
        self.current_rollout_id: int = 0  # set by generate_rollout_async

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        multistep_k = get_effective_k(self.args, self.current_rollout_id)
        for group in samples:
            if multistep_k > 0:
                # Multi-step refinement: k sequential edit→evaluate→feedback loops
                gen_coro = generate_and_rm_multistep_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
            else:
                # Standard single-step generation
                gen_coro = generate_and_rm_group(
                    self.args,
                    group,
                    sampling_params=self.sampling_params.copy(),
                    evaluation=False,
                )
            self.pendings.add(asyncio.create_task(gen_coro))
        self.remaining_batch_size += len(samples)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    # Process prompt to create text and image payload
    image_data = []
    if isinstance(sample.prompt, str):
        text_prompt = sample.prompt
    else:  # Multimodal prompt (list of dicts)
        text_prompt = ""
        # sglang uses a placeholder to insert image features
        image_token = state.tokenizer.special_tokens_map.get("image_token", "<image>")
        for part in sample.prompt:
            if part["type"] == "text":
                text_prompt += part["text"]
            elif part["type"] == "image":
                text_prompt += image_token
                try:
                    with Image.open(part["path"]) as image:
                        buffer = io.BytesIO()
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        image.save(buffer, format="JPEG")
                        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        image_data.append(img_b64)
                except Exception as e:
                    print(f"Error processing image {part['path']}: {e}")
                    sample.status = Sample.Status.ABORTED
                    return sample

    if len(sample.response) > 0:
        # Adjust max_new_tokens for subsequent generation turns
        prompt_len = len(state.tokenizer(text_prompt, add_special_tokens=False)["input_ids"])
        sampling_params["max_new_tokens"] -= len(sample.tokens) - prompt_len

    if sampling_params["max_new_tokens"] <= 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        payload["image_data"] = image_data

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        prompt_token_ids = state.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
        payload["input_ids"] = prompt_token_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_token_ids

    output = await post(url, payload)

    # Extract new response tokens
    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_response_tokens, new_response_log_probs = [], []

    # Update sample with tokens directly - avoiding re-tokenization
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]
    if "weight_version" in output["meta_info"]:
        sample.weight_versions.append(output["meta_info"]["weight_version"])
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def generate_and_rm(
    args: Namespace,
    sample: Union[Sample, list[Sample]],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Union[Sample, list[Sample]]:
    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        if args.custom_generate_function_path is not None:
            custom_generate_func = load_function(args.custom_generate_function_path)
            sample = await custom_generate_func(args, sample, sampling_params)
        else:
            sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample

        sample.reward = await async_rm(args, sample)

    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will not do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards):
            sample.reward = reward

    return group


# ---------------------------------------------------------------------------
# Multi-step refinement: k sequential edit→evaluate→feedback loops per sample
# ---------------------------------------------------------------------------
# Shared prompt/feedback logic lives in evolve_prompt_utils.py (single source
# of truth used by both training and inference-only paths).
from slime.rollout.evolve_prompt_utils import (
    FULL_PROGRAM_INSTRUCTIONS as _FULL_PROGRAM_INSTRUCTIONS,
    strip_traceback_boilerplate as _strip_traceback_boilerplate,
    strip_boilerplate_from_code_blocks as _strip_boilerplate_from_code_blocks,
    rewrite_prompt_for_full_program as _rewrite_prompt_for_full_program,
    format_multistep_feedback as _format_multistep_feedback,
    extract_evolve_block as _extract_evolve_block,
)

async def _generate_and_rm_multistep_single(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    k: int,
) -> Sample:
    """Run k sequential refinement steps for one sample.

    After each generation the response is evaluated (via evolving_gym_rm).
    A chat-template feedback turn is injected between steps so the model
    can observe the evaluation result before generating the next edit.

    Uses **full-program mode**: the model outputs the complete EVOLVE-BLOCK
    code as a Python code block, not a SEARCH/REPLACE diff.

    The final reward is sum(raw_scores[1:]) — proportional scaling of
    ThetaEvolve's per-step evaluator signal.
    """
    from .rm_hub.evolving_gym_rm import compute_multistep_reward

    state = GenerateState(args)
    tokenizer = state.tokenizer

    # ── Detect language from initial program file extension ──
    _init_prog = getattr(args, "evolving_gym_initial_program", "") or ""
    _lang = "cpp" if _init_prog.endswith((".cpp", ".cc", ".cxx", ".c")) else "python"

    # ── Rewrite prompt: diff → full-program mode ──
    original_prompt_text = sample.prompt
    rewritten_prompt_text = _rewrite_prompt_for_full_program(
        original_prompt_text,
        single_reference=getattr(args, "single_reference", False),
        language=_lang,
    )
    if rewritten_prompt_text != original_prompt_text:
        sample.prompt = rewritten_prompt_text
        # Re-tokenize so sample.tokens matches the rewritten prompt
        new_token_ids = tokenizer(rewritten_prompt_text, add_special_tokens=False)["input_ids"]
        sample.tokens = new_token_ids
        print(
            f"[MultiStep] sample={sample.index} rewrote prompt for full-program mode "
            f"(tokens: {len(sample.tokens)}, lang={_lang})"
        )

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["_language"] = _lang

    # ── OpenAI path: rewrite the user content in _chat_messages too ──
    if getattr(args, "openai_api_base", None) and "_chat_messages" in sample.metadata:
        for msg in sample.metadata["_chat_messages"]:
            if msg["role"] == "user":
                msg["content"] = _rewrite_prompt_for_full_program(
                    msg["content"],
                    single_reference=getattr(args, "single_reference", False),
                    language=_lang,
                )
                break  # only the first user message is the prompt

    # Debug flag: dump full text for the first sample in each rollout group.
    # sample.index is global (0,1,...15,16,...31,...), so we use modulo to detect
    # the first sample of each group of n_samples_per_prompt.
    _n_per_group = getattr(args, "n_samples_per_prompt", 16) or 16
    _dbg = (sample.index % _n_per_group == 0)
    _save_dir = getattr(args, "save", None) or "."
    import os as _os
    _os.makedirs(_save_dir, exist_ok=True)
    _dbg_file = _os.path.join(_save_dir, "debug_trace_sample0.txt")

    def _dbg_write(text: str):
        """Write debug text to both stdout and the debug trace file."""
        print(text, flush=True)
        with open(_dbg_file, "a") as f:
            f.write(text + "\n")

    if _dbg:
        # Append a separator for each new rollout (no longer clear the file)
        with open(_dbg_file, "a") as f:
            f.write(f"\n\n{'#'*80}\n# NEW ROLLOUT — sample.index={sample.index}\n{'#'*80}\n\n")
        _dbg_write(
            f"\n{'='*80}\n"
            f"[DEBUG SAMPLE 0] ══ INITIAL PROMPT (after full-program rewrite) ══\n"
            f"{'='*80}\n"
            f"{sample.prompt}\n"
            f"{'='*80}\n"
            f"[DEBUG SAMPLE 0] prompt tokens={len(sample.tokens)}\n"
            f"{'='*80}\n"
        )

    # Track per-step data.  Index 0 = parent (before any edit).
    # raw_scores:  raw evaluator scores (including negative error codes like -0.1..-0.4)
    # validities:  True/False per entry
    # error_codes: original combined_score for invalid steps (e.g. -0.2), None for valid
    # _step_traces: structured per-step debug info for the checkpoint trace file
    _step_traces: list[dict] = []
    parent_program = (sample.metadata or {}).get("parent_program")
    parent_score = 0.0
    if parent_program and hasattr(parent_program, "metrics") and parent_program.metrics:
        parent_score = parent_program.metrics.get("combined_score", 0.0)
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["_original_parent_score"] = parent_score
    # Per-sample trajectory context for evaluator lazy detection.
    # We keep only current k-step chain, not global history.
    seen_codes: list[str] = []

    def _append_seen_code(codes: list[str], code: str, max_len: int) -> None:
        if not code:
            return
        if code not in codes:
            codes.append(code)
        # Keep only the most recent max_len context entries to enforce bounded context.
        if len(codes) > max_len:
            del codes[: len(codes) - max_len]

    # ── Cross-rollout trajectory history ──
    # Each entry: {"response": str, "feedback": str}
    # Full model CoT + evaluator feedback per step, injected as chat turns
    # for context continuity (the model sees its own previous generations).
    context_horizon = getattr(args, "context_horizon", 0)
    trajectory_history: list[dict] = []
    if context_horizon > 0 and parent_program is not None:
        parent_meta = getattr(parent_program, "metadata", {}) or {}
        inherited_hist = parent_meta.get("_trajectory_history", [])
        _raw_hist_len = len(inherited_hist) if isinstance(inherited_hist, list) else 0
        if isinstance(inherited_hist, list):
            trajectory_history = [
                h for h in inherited_hist
                if isinstance(h, dict) and "response" in h
            ]
        _filtered_hist_len = len(trajectory_history)
        # Truncate to context_horizon
        if len(trajectory_history) > context_horizon:
            trajectory_history = trajectory_history[-context_horizon:]
        print(
            f"[TrajLoad] sample={sample.index} "
            f"parent_id={getattr(parent_program, 'id', 'N/A')[:8] if parent_program else 'none'} "
            f"raw_entries={_raw_hist_len} valid={_filtered_hist_len} "
            f"after_trunc={len(trajectory_history)} horizon={context_horizon}",
            flush=True,
        )

    if parent_program is not None:
        parent_code = getattr(parent_program, "code", "") or ""
        parent_meta = getattr(parent_program, "metadata", {}) or {}
        inherited = parent_meta.get("_multistep_seen_codes", [])
        if isinstance(inherited, list):
            seen_codes.extend([c for c in inherited if isinstance(c, str) and c])
        _append_seen_code(seen_codes, parent_code, k)
        assert len(seen_codes) <= k, f"_multistep_seen_codes too long at init: len={len(seen_codes)} k={k}"
        if not hasattr(parent_program, "metadata") or parent_program.metadata is None:
            parent_program.metadata = {}
        parent_program.metadata["_multistep_seen_codes"] = list(seen_codes)

    raw_scores: list[float] = [parent_score]
    validities: list[bool] = [True]           # parent is always valid
    error_codes: list[float | None] = [None]  # parent has no error
    step_resp_lens: list[int] = []            # resp_len after each refinement step

    # loss_mask and rollout_log_probs accumulators
    loss_mask: list[int] = []
    all_log_probs: list[float] = []
    cumulative_response_len = 0
    prev_sample_lp_count = 0
    best_score = parent_score

    # ── Target score for feedback display (best-known value for this task) ──
    _target_score: Optional[float] = getattr(args, "target_score", None)

    # ── Adaptive-k: early termination when model stops making progress ──
    # If adaptive_k_patience > 0, terminate the chain after this many
    # steps (consecutive or cumulative, per adaptive_k_patience_mode) that do
    # NOT beat the current database worst-score.
    _adaptive_k_patience: int = max(0, getattr(args, "adaptive_k_patience", 0))
    _adaptive_k_patience_mode: str = getattr(args, "adaptive_k_patience_mode", "consecutive")
    _no_improve_streak: int = 0   # consecutive bad steps (resets on good step)
    _no_improve_total: int = 0    # cumulative bad steps (never resets)
    _step_scores_history: list = []  # scores of completed steps, for feedback display

    _hist_len = len(trajectory_history) if context_horizon > 0 else 0
    print(
        f"[MultiStep START] sample={sample.index} k={k} parent_score={parent_score:.4f} "
        f"seen_codes_init={len(seen_codes)} traj_hist={_hist_len}/{context_horizon} "
        f"parent_id={getattr(parent_program, 'id', 'N/A')[:8] if parent_program else 'none'}",
        flush=True,
    )

    # ── Inject cross-rollout trajectory history as chat turns ──
    # Makes the model see its own previous CoT + feedback as if it was
    # continuously generating, rather than starting fresh each rollout.
    if trajectory_history and context_horizon > 0:
        # Detect assistant end-of-turn text for the chat format
        _eot_text = ""
        try:
            _dummy = tokenizer.apply_chat_template(
                [{"role": "assistant", "content": "X"}],
                tokenize=False, add_generation_prompt=False,
            )
            _idx = _dummy.rfind("X")
            if _idx >= 0:
                _eot_text = _dummy[_idx + 1:].rstrip("\n")
        except Exception:
            _eot_text = tokenizer.eos_token or ""

        _injected_tokens = 0
        _injected_prompt_ext = ""  # extend sample.prompt so generate() doesn't penalise max_new_tokens
        _ctx_before = len(sample.tokens)

        for h_idx, h_entry in enumerate(trajectory_history):
            h_resp = h_entry.get("response", "")
            h_fb = h_entry.get("feedback", "")
            if not h_resp:
                print(f"[TrajInject] sample={sample.index} turn {h_idx+1}: SKIP (empty response)", flush=True)
                continue

            # 1. Inject historical assistant response (generation prompt already in tokens)
            # Must end with exactly the EOT marker (<|im_end|>) and nothing else,
            # so the following feedback turn starts immediately — matching continuous gen.
            h_resp_clean = h_resp.rstrip()
            if _eot_text and not h_resp_clean.endswith(_eot_text):
                h_resp_inject = h_resp_clean + _eot_text
            else:
                h_resp_inject = h_resp_clean

            h_resp_tokens = tokenizer(h_resp_inject, add_special_tokens=False)["input_ids"]
            sample.tokens = sample.tokens + h_resp_tokens
            sample.response += h_resp_inject
            sample.response_length += len(h_resp_tokens)
            cumulative_response_len += len(h_resp_tokens)
            loss_mask.extend([0] * len(h_resp_tokens))
            all_log_probs.extend([0.0] * len(h_resp_tokens))
            _injected_tokens += len(h_resp_tokens)
            _injected_prompt_ext += h_resp_inject

            _fb_tok_count = 0
            # 2. Inject historical user feedback (same format as normal k-step feedback)
            if h_fb:
                fb_messages = [{"role": "user", "content": h_fb}]
                _chat_template_kwargs = (
                    {"enable_thinking": args.enable_thinking}
                    if getattr(args, "enable_thinking", None) is not None
                    else {}
                )
                fb_str = tokenizer.apply_chat_template(
                    fb_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **_chat_template_kwargs,
                )
                fb_token_ids = tokenizer(fb_str, add_special_tokens=False)["input_ids"]
                sample.tokens = sample.tokens + fb_token_ids
                sample.response += fb_str
                sample.response_length += len(fb_token_ids)
                cumulative_response_len += len(fb_token_ids)
                loss_mask.extend([0] * len(fb_token_ids))
                all_log_probs.extend([0.0] * len(fb_token_ids))
                _injected_tokens += len(fb_token_ids)
                _injected_prompt_ext += fb_str
                _fb_tok_count = len(fb_token_ids)

            _fb_preview = repr(h_fb[:80]) + "..." if len(h_fb) > 80 else repr(h_fb)
            print(
                f"[TrajInject] sample={sample.index} turn {h_idx+1}/{len(trajectory_history)}: "
                f"resp={len(h_resp_tokens)}tok  fb={_fb_tok_count}tok  "
                f"feedback={_fb_preview}",
                flush=True,
            )

        # Extend sample.prompt so generate() counts injected tokens as "prompt"
        # and doesn't reduce max_new_tokens for them.
        sample.prompt += _injected_prompt_ext

        print(
            f"[TrajInject DONE] sample={sample.index} "
            f"injected {len(trajectory_history)} history turns  "
            f"tokens: {_injected_tokens} injected + {_ctx_before} prompt = {len(sample.tokens)} total  "
            f"loss_mask: {sum(loss_mask)} trainable / {len(loss_mask)} total",
            flush=True,
        )
        if _dbg:
            _dbg_write(
                f"\n[DEBUG SAMPLE 0] Injected {len(trajectory_history)} trajectory history turns "
                f"({_injected_tokens} tokens, total_ctx={len(sample.tokens)})\n"
            )

    # ── per-rollout trajectory file: one file per debug sample so parallel
    # chains never overwrite each other.  Files land in <save>/trajectories/.
    _traj_dir = _os.path.join(_save_dir, "trajectories")
    if _dbg:
        _os.makedirs(_traj_dir, exist_ok=True)
    _traj_dbg_file = (
        _os.path.join(_traj_dir, f"traj_{sample.index:06d}.txt") if _dbg else None
    )
    if _dbg:
        with open(_traj_dbg_file, "w") as _tf:
            _tf.write(f"{'='*80}\n")
            _tf.write(f"TRAJECTORY — sample={sample.index}  k={k}  "
                       f"parent_score={parent_score:.4f}  "
                       f"parent_id={getattr(parent_program, 'id', 'N/A')[:8] if parent_program else 'none'}\n")
            _tf.write(f"{'='*80}\n\n")
            _tf.write(f"[PROMPT] ({len(sample.prompt)} chars)\n")
            _tf.write(f"{'─'*80}\n")
            _tf.write(sample.prompt)
            _tf.write(f"\n{'─'*80}\n")

    _openai_mode = getattr(args, "openai_api_base", None) is not None
    if _openai_mode:
        from slime.rollout.openai_generate import generate_openai as _generate_fn
    else:
        _generate_fn = generate

    for step_idx in range(k):
        # Remember response text before this generation step
        response_before = sample.response

        # ── Generate ──
        async with state.semaphore:
            if state.aborted:
                sample.status = Sample.Status.ABORTED
                break
            sample = await _generate_fn(args, sample, sampling_params.copy())

        if sample.status == Sample.Status.ABORTED:
            break

        # Mark this step's LLM-generated tokens in loss_mask
        new_llm_tokens = sample.response_length - cumulative_response_len
        loss_mask.extend([1] * new_llm_tokens)
        if sample.rollout_log_probs is not None:
            new_lps = sample.rollout_log_probs[prev_sample_lp_count:]
            all_log_probs.extend(new_lps)
            prev_sample_lp_count = len(sample.rollout_log_probs)
        cumulative_response_len = sample.response_length

        # Extract ONLY this step's new response text
        new_response_text = sample.response[len(response_before):]

        if _dbg:
            _dbg_write(
                f"\n{'─'*80}\n"
                f"[DEBUG SAMPLE 0] ── STEP {step_idx+1}/{k} GENERATION ({len(new_response_text)} chars) ──\n"
                f"{'─'*80}\n"
                f"{new_response_text}\n"
                f"{'─'*80}\n"
                f"[DEBUG SAMPLE 0] step {step_idx+1} generation tokens={new_llm_tokens} "
                f"cumulative_resp_len={cumulative_response_len}\n"
                f"{'─'*80}"
            )
            # Also write to trajectory_debug.txt (step-by-step)
            with open(_traj_dbg_file, "a") as _tf:
                _tf.write(f"\n\n[STEP {step_idx+1}/{k} → OUTPUT] ({len(new_response_text)} chars)\n")
                _tf.write(f"{'─'*80}\n")
                _tf.write(new_response_text)
                _tf.write(f"\n{'─'*80}\n")

        # ── Evaluate via RM using a temporary sample containing only the new response ──
        eval_metadata = copy.deepcopy(sample.metadata) if sample.metadata else {}
        # N-repeat only at final step (intermediate steps use N=1 for fast feedback)
        _env_n_repeat = int(_os.environ.get("LEARNX_EVAL_N_REPEAT", "1"))
        eval_metadata["_eval_n_repeat"] = _env_n_repeat if (step_idx == k - 1) else 1
        eval_sample = Sample(
            index=sample.index,
            prompt=sample.prompt,
            response=new_response_text,
            metadata=eval_metadata,
        )
        # Snapshot db_worst BEFORE evaluation so the child's own score
        # doesn't pollute the threshold (RM adds child to database immediately).
        _db_worst_snapshot = get_database_worst_score()
        reward_info = await async_rm(args, eval_sample)
        step_metrics = reward_info.get("metrics", {}) if isinstance(reward_info, dict) else {}
        # Extract error message: prefer artifacts (contains detailed validation/execution errors)
        step_artifacts = reward_info.get("artifacts", {}) if isinstance(reward_info, dict) else {}
        step_error = None
        if isinstance(step_artifacts, dict) and step_artifacts.get("error"):
            step_error = step_artifacts["error"]
        elif step_metrics and step_metrics.get("error"):
            step_error = step_metrics["error"]
        # Strip boilerplate frames from tracebacks so the model doesn't see
        # framework internals like run_circle_packing() or save_search_results()
        # Also cap error length to prevent feedback token explosion (→ OOM).
        _MAX_ERROR_CHARS = int(_os.environ.get("LEARNX_MAX_ERR_CHARS", "10000"))
        if step_error:
            step_error = _strip_traceback_boilerplate(step_error)
            if len(step_error) > _MAX_ERROR_CHARS:
                print(
                    f"[MultiStep] ⚠ sample={sample.index} step={step_idx+1}/{k} "
                    f"error truncated: {len(step_error)} → {_MAX_ERROR_CHARS} chars",
                    flush=True,
                )
                step_error = step_error[:_MAX_ERROR_CHARS] + f"\n... (truncated)"

        # Extract raw evaluator score and validity (SkyAgent style)
        step_combined_raw = step_metrics.get("combined_score", 0.0) if step_metrics else 0.0
        step_validity = step_metrics.get("validity", None) if step_metrics else None
        # Determine if this step is valid: use explicit validity flag if available,
        # otherwise fall back to checking if combined_score is an error code (< 0)
        if step_validity is not None:
            step_is_valid = float(step_validity) > 0.5
        else:
            step_is_valid = step_combined_raw >= 0
        raw_scores.append(float(step_combined_raw))
        validities.append(step_is_valid)
        error_codes.append(step_combined_raw if not step_is_valid else None)
        _prev_best = best_score
        best_score = max(best_score, float(step_combined_raw))

        # ── Periodic inner-step wandb log (every N evals) ──
        _inner_step_wandb_log(args, step_idx, k, float(step_combined_raw), step_is_valid)

        # ── Adaptive-k: track steps that do NOT improve on the chain's own best ──
        # A step "contributes" if its score strictly beats the best score seen so
        # far in THIS chain (_prev_best, updated just above).  This measures whether
        # local refinement is making real progress, independent of the DB state.
        _db_worst = _db_worst_snapshot  # kept for logging only
        if step_is_valid and float(step_combined_raw) > _prev_best:
            _no_improve_streak = 0  # new chain-best → reset consecutive counter
            if _adaptive_k_patience > 0:
                _bad_count = _no_improve_total if _adaptive_k_patience_mode == "cumulative" else _no_improve_streak
                print(
                    f"[AdaptiveK] sample={sample.index} step={step_idx+1}/{k} "
                    f"score={step_combined_raw:.4f} > chain_best={_prev_best:.4f} ✓ "
                    f"{'streak reset' if _adaptive_k_patience_mode == 'consecutive' else f'cumulative_bad={_no_improve_total}'}",
                    flush=True,
                )
        else:
            _no_improve_streak += 1
            _no_improve_total += 1
            _reason = (
                f"invalid(score={step_combined_raw:.4f})" if not step_is_valid
                else f"score={step_combined_raw:.4f} <= chain_best={_prev_best:.4f}"
            )
            if _adaptive_k_patience > 0:
                if _adaptive_k_patience_mode == "cumulative":
                    print(
                        f"[AdaptiveK] sample={sample.index} step={step_idx+1}/{k} "
                        f"no improve ({_reason})  cumulative_bad={_no_improve_total}/{_adaptive_k_patience}",
                        flush=True,
                    )
                else:
                    print(
                        f"[AdaptiveK] sample={sample.index} step={step_idx+1}/{k} "
                        f"no improve ({_reason})  streak={_no_improve_streak}/{_adaptive_k_patience}",
                        flush=True,
                    )

        child_program = reward_info.get("child_program") if isinstance(reward_info, dict) else None
        if child_program is not None and getattr(child_program, "code", ""):
            child_code = getattr(child_program, "code", "") or ""
            _append_seen_code(seen_codes, child_code, k)
            assert len(seen_codes) <= k, (
                f"_multistep_seen_codes too long at step={step_idx+1}: len={len(seen_codes)} k={k}"
            )
            if not hasattr(child_program, "metadata") or child_program.metadata is None:
                child_program.metadata = {}
            child_program.metadata["_multistep_seen_codes"] = list(seen_codes)

            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata["parent_program"] = child_program

        step_resp_lens.append(sample.response_length)

        if _dbg:
            # Filter internal/training metrics that would confuse readers
            _METRICS_EXCLUDE = {"rl_normalized_reward", "eval_time", "program_runtime",
                                 "program_timeout", "program_exit_code"}
            _clean_metrics = {k_: v_ for k_, v_ in (step_metrics or {}).items()
                              if k_ not in _METRICS_EXCLUDE}
            _dbg_write(
                f"\n[DEBUG SAMPLE 0] step {step_idx+1}/{k} "
                f"raw={step_combined_raw:.4f} valid={step_is_valid} best={best_score:.4f} "
                f"error={step_error} metrics={_clean_metrics}"
            )
            # Append score line to trajectory_debug.txt
            _err_brief = f"  error={step_error}" if step_error else ""
            with open(_traj_dbg_file, "a") as _tf:
                _tf.write(f">>> score={step_combined_raw:.4f}  valid={step_is_valid}  best={best_score:.4f}{_err_brief}\n")

        # Classify error type for quick log scanning
        # Error codes: -0.1=VALIDATION_FAILED  -0.2=EXECUTION_ERROR
        #              -0.3=NO_VALID_CHANGE    -0.4=NO_CODE_BLOCK
        _err_tag = ""
        if not step_is_valid:
            _ec = float(step_combined_raw)
            if abs(_ec - (-0.1)) < 0.01:
                _err_tag = " [VAL_FAILED]"
            elif abs(_ec - (-0.2)) < 0.01:
                _err_tag = " [EXEC_ERR]"
            elif abs(_ec - (-0.3)) < 0.01:
                _err_tag = " [LAZY/NO_CHANGE]"
            elif abs(_ec - (-0.4)) < 0.01:
                _err_tag = " [NO_CODE_BLOCK]"
            else:
                _err_tag = f" [INVALID={_ec}]"
        print(
            f"[MultiStep] sample={sample.index} step={step_idx+1}/{k} "
            f"raw={step_combined_raw:.4f} valid={step_is_valid}{_err_tag} "
            f"seen_ctx={len(seen_codes)}/{k} best={best_score:.4f} "
            f"resp_len={sample.response_length} "
            f"parent_propagated={'yes' if child_program is not None else 'NO_CHILD⚠'}",
            flush=True,
        )

        # ── Structured trace record for checkpoint debug output ──
        _parsed_code = getattr(child_program, "code", "") if child_program else ""
        _step_traces.append({
            "sample": sample.index,
            "step": step_idx + 1,
            "k": k,
            "raw_score": float(step_combined_raw),
            "valid": step_is_valid,
            "err_tag": _err_tag.strip(),
            "parsed_code_len": len(_parsed_code) if _parsed_code else 0,
            "parsed_code_head": (_parsed_code[:300] if _parsed_code else ""),
            "parsed_code_tail": (_parsed_code[-200:] if _parsed_code and len(_parsed_code) > 500 else ""),
            "response_len_chars": len(new_response_text),
            "error_msg": (step_error[:300] if step_error else None),
        })

        # Generate feedback: two variants
        # 1) For current rollout injection — includes "step X/Y, Z remaining"
        feedback_current = _format_multistep_feedback(
            step=step_idx + 1,
            total_steps=k,
            score=float(step_combined_raw),
            best_score=best_score,
            metrics=step_metrics,
            error=step_error,
            chain_best_score=best_score,
            target_score=_target_score,
            db_worst_score=_db_worst,
            no_improve_streak=_no_improve_streak,
            adaptive_k_patience=_adaptive_k_patience,
            parent_score=parent_score,
            step_scores_history=_step_scores_history,
            language=_lang,
        )
        # 2) For trajectory history — neutral "[Attempt N]" to avoid confusing
        #    the model when step counts reset across rollouts.
        _cum_attempt = len(trajectory_history) + 1  # 1-based, before append
        feedback_for_hist = _format_multistep_feedback(
            step=step_idx + 1,
            total_steps=k,
            score=float(step_combined_raw),
            best_score=best_score,
            metrics=step_metrics,
            error=step_error,
            cumulative_attempt=_cum_attempt,
            chain_best_score=best_score,
            target_score=_target_score,
            db_worst_score=_db_worst,
            no_improve_streak=_no_improve_streak,
            adaptive_k_patience=_adaptive_k_patience,
            parent_score=parent_score,
            step_scores_history=_step_scores_history,
            language=_lang,
        )
        # Record this step's score for future feedback display
        _step_scores_history = _step_scores_history + [float(step_combined_raw)]

        # ── Update cross-rollout trajectory history (full CoT + feedback) ──
        if context_horizon > 0:
            trajectory_history.append({
                "response": new_response_text,
                "feedback": feedback_for_hist,
            })
            if len(trajectory_history) > context_horizon:
                trajectory_history = trajectory_history[-context_horizon:]
            # Store in child_program metadata so it persists in the database
            if child_program is not None:
                if not hasattr(child_program, "metadata") or child_program.metadata is None:
                    child_program.metadata = {}
                child_program.metadata["_trajectory_history"] = list(trajectory_history)
            print(
                f"[TrajHist] sample={sample.index} step={step_idx+1}/{k} "
                f"stored attempt #{_cum_attempt} "
                f"resp_len={len(new_response_text)} chars "
                f"hist_size={len(trajectory_history)}/{context_horizon}",
                flush=True,
            )

        # If truncated, stop early
        if sample.status == Sample.Status.TRUNCATED:
            if _dbg:
                _dbg_write(f"[DEBUG SAMPLE 0] TRUNCATED at step {step_idx+1}/{k}")
            break

        # ── Adaptive-k: early termination when progress stalls ──
        if _adaptive_k_patience > 0 and step_idx < k - 1:
            _bad_count = _no_improve_total if _adaptive_k_patience_mode == "cumulative" else _no_improve_streak
            if _bad_count >= _adaptive_k_patience:
                _db_worst_log = f"{_db_worst:.4f}" if _db_worst is not None else "N/A"
                # Build per-step score summary for diagnosis (✓ = improved over previous chain best)
                _step_scores_fmt = "  ".join(
                    f"s{i+1}={s:.4f}{'✓' if i == 0 and s > parent_score else ('✓' if i > 0 and s > max(raw_scores[1:i+1]) else '✗')}"
                    for i, s in enumerate(raw_scores[1:])  # skip parent at index 0
                )
                _stop_reason = (
                    f"{_no_improve_total} cumulative steps not improving chain_best={best_score:.4f}"
                    if _adaptive_k_patience_mode == "cumulative"
                    else f"{_no_improve_streak} consecutive steps not improving chain_best={best_score:.4f}"
                )
                print(
                    f"[AdaptiveK] ⛔ sample={sample.index} EARLY STOP step={step_idx+1}/{k}  "
                    f"reason: {_stop_reason}  "
                    f"chain_best={best_score:.4f}  parent={parent_score:.4f}  "
                    f"per_step_scores=[{_step_scores_fmt}]",
                    flush=True,
                )
                if _dbg:
                    _dbg_write(
                        f"[DEBUG SAMPLE 0] AdaptiveK EARLY STOP step={step_idx+1}/{k}  "
                        f"mode={_adaptive_k_patience_mode}  bad_count={_bad_count}/{_adaptive_k_patience}  "
                        f"db_worst={_db_worst_log}  chain_best={best_score:.4f}  "
                        f"per_step=[{_step_scores_fmt}]"
                    )
                break

        # ── If not last step, inject feedback turn ──
        if step_idx < k - 1:
            feedback_text = feedback_current  # use the step-aware variant for current rollout

            if _dbg:
                _dbg_write(
                    f"\n{'─'*80}\n"
                    f"[DEBUG SAMPLE 0] ── STEP {step_idx+1}/{k} FEEDBACK (injected) ──\n"
                    f"{'─'*80}\n"
                    f"{feedback_text}\n"
                    f"{'─'*80}"
                )
                # Also write feedback to trajectory_debug.txt
                with open(_traj_dbg_file, "a") as _tf:
                    _tf.write(f"\n[STEP {step_idx+1}/{k} → FEEDBACK]\n")
                    _tf.write(f"{'─'*80}\n")
                    _tf.write(feedback_text)
                    _tf.write(f"\n{'─'*80}\n")

            feedback_messages = [
                {"role": "user", "content": feedback_text},
            ]
            _chat_template_kwargs = (
                {"enable_thinking": args.enable_thinking}
                if getattr(args, "enable_thinking", None) is not None
                else {}
            )
            feedback_str = tokenizer.apply_chat_template(
                feedback_messages,
                tokenize=False,
                add_generation_prompt=True,
                **_chat_template_kwargs,
            )
            feedback_token_ids = tokenizer(feedback_str, add_special_tokens=False)["input_ids"]

            # Safety: cap feedback tokens to prevent OOM from huge error messages.
            # Normal feedback ≈ 100-200 tokens; anything above cap means the error text
            # wasn't properly truncated upstream.
            _MAX_FEEDBACK_TOKENS = int(_os.environ.get("LEARNX_MAX_FEEDBACK_TOKENS", "2048"))
            if len(feedback_token_ids) > _MAX_FEEDBACK_TOKENS:
                print(
                    f"[MultiStep] ⚠ sample={sample.index} step={step_idx+1}/{k} "
                    f"feedback tokens={len(feedback_token_ids)} > {_MAX_FEEDBACK_TOKENS}, "
                    f"truncating to prevent OOM",
                    flush=True,
                )
                feedback_token_ids = feedback_token_ids[:_MAX_FEEDBACK_TOKENS]
                # Re-decode to keep feedback_str consistent with tokens
                feedback_str = tokenizer.decode(feedback_token_ids, skip_special_tokens=False)

            # Append feedback tokens to the sample
            sample.tokens = sample.tokens + feedback_token_ids
            sample.response += feedback_str
            sample.response_length += len(feedback_token_ids)
            cumulative_response_len += len(feedback_token_ids)

            # Feedback tokens: loss_mask=0, log_prob=0.0 (not LLM-generated)
            loss_mask.extend([0] * len(feedback_token_ids))
            all_log_probs.extend([0.0] * len(feedback_token_ids))

            # ── OpenAI path: append feedback as user message ──
            if _openai_mode and sample.metadata and "_chat_messages" in sample.metadata:
                sample.metadata["_chat_messages"].append(
                    {"role": "user", "content": feedback_text}
                )

            if _dbg:
                _dbg_write(
                    f"[DEBUG SAMPLE 0] feedback tokens={len(feedback_token_ids)} "
                    f"cumulative_resp_len={cumulative_response_len}"
                )

            # Reset status so generate() will continue
            sample.status = Sample.Status.PENDING

    # ── Compute final multi-step reward ──
    reward_agg = getattr(args, "multistep_reward_agg", "sum")
    accumulated_reward = compute_multistep_reward(raw_scores, agg=reward_agg, validities=validities)
    step_agg_dbg = accumulated_reward
    n_invalid_dbg = sum(1 for v in validities[1:] if not v)
    n_lazy_dbg = sum(1 for ec in error_codes[1:] if ec is not None and abs(float(ec) - (-0.3)) < 0.01)
    n_steps_done = len(raw_scores) - 1
    print(
        f"[Reward] sample={sample.index} R={reward_agg}(raw[1:])={step_agg_dbg:.4f} "
        f"steps_done={n_steps_done}/{k} invalid={n_invalid_dbg} lazy={n_lazy_dbg} "
        f"per_step={[round(s,4) for s in raw_scores[1:]]} "
        f"=> reward={accumulated_reward:.4f}",
        flush=True,
    )

    reward_key = getattr(args, "reward_key", "reward")
    # Capture original parent ID (metadata["parent_program"] may be overwritten in multistep)
    _orig_parent_id = None
    if parent_program and hasattr(parent_program, "id"):
        _orig_parent_id = parent_program.id
    reward_dict = {
        reward_key: accumulated_reward,
        "multistep_raw_scores": raw_scores,   # raw evaluator scores (includes error codes)
        "multistep_validities": validities,   # [True, False, True, ...] per step
        "multistep_error_codes": error_codes, # [None, -0.2, None, ...] original error code if invalid
        "multistep_resp_lens": step_resp_lens,
        # Actual refinement steps taken (may be < k when adaptive-k fires early).
        # Used by rollout.py to compute cum_actual_evals for fair x-axis across k values.
        "chain_actual_k": n_steps_done,
        # Original parent program ID (stable across multistep; used by improvement-rate eviction classify)
        "original_parent_id": _orig_parent_id,
    }
    if isinstance(sample.reward, dict):
        sample.reward.update(reward_dict)
    else:
        sample.reward = reward_dict

    # Store structured step traces in metadata for checkpoint debug output
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["_step_traces"] = _step_traces

    # Set loss_mask and log_probs
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = all_log_probs

    _lm_ones = sum(loss_mask)
    _lm_total = len(loss_mask)
    _lm_ratio = _lm_ones / _lm_total if _lm_total > 0 else 0.0
    print(
        f"[MultiStep DONE] sample={sample.index} "
        f"raw_scores={[round(s,4) for s in raw_scores]} "
        f"valid={validities[1:]} errors={[ec for ec in error_codes[1:]]} "
        f"reward={accumulated_reward:.4f} "
        f"loss_mask={_lm_ones}/{_lm_total} ({_lm_ratio:.1%} LLM tokens)",
        flush=True,
    )

    if _dbg:
        _dbg_write(
            f"\n{'='*80}\n"
            f"[DEBUG SAMPLE 0] ══ FINAL SUMMARY ══\n"
            f"  raw_scores  = {[round(s,4) for s in raw_scores]}\n"
            f"  validities  = {validities}\n"
            f"  error_codes = {error_codes}\n"
            f"  resp_lens   = {step_resp_lens}\n"
            f"  reward      = {accumulated_reward:.4f}\n"
            f"  loss_mask   = {sum(loss_mask)}/{len(loss_mask)} (ones/total)\n"
            f"  parent_score= {parent_score:.4f}  best_score={best_score:.4f}\n"
            f"  traj_hist   = {len(trajectory_history)} entries\n"
            f"{'='*80}\n"
        )
        # Append final summary to trajectory_debug.txt
        if _traj_dbg_file:
            with open(_traj_dbg_file, "a") as _tf:
                _tf.write(f"\n\n{'='*80}\n")
                _tf.write(f"[SUMMARY] reward={accumulated_reward:.4f}  "
                           f"raw_scores={[round(s,4) for s in raw_scores]}  "
                           f"best={best_score:.4f}  parent={parent_score:.4f}\n")
                _tf.write(f"{'='*80}\n")
            print(f"[TrajDebug] Wrote {_traj_dbg_file}", flush=True)

    return sample


async def generate_and_rm_multistep_group(
    args: Namespace,
    group: list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample]:
    """Multi-step refinement variant of generate_and_rm_group.

    Each sample in the group independently goes through k refinement steps.
    All K siblings run in parallel (but each sibling's steps are sequential).
    """
    state = GenerateState(args)
    if state.aborted:
        return group

    k = get_effective_k(args, state.current_rollout_id)
    print(
        f"[MultiStep config] rollout={state.current_rollout_id} k={k} "
        f"x_advance=+{get_rollout_x_advance(args, state.current_rollout_id)} "
        f"inner_cum={get_inner_step_cum(args, state.current_rollout_id)}",
        flush=True,
    )

    tasks = []
    for sample in group:
        tasks.append(
            _generate_and_rm_multistep_single(
                args, sample, sampling_params.copy(), k,
            )
        )

    group = await asyncio.gather(*tasks)

    # Group summary: aggregate rewards for quick sanity check
    rewards = []
    for s in group:
        r = getattr(s, "reward", None)
        if isinstance(r, dict):
            rewards.append(r.get(getattr(args, "reward_key", "reward"), 0.0))
        elif isinstance(r, (int, float)):
            rewards.append(float(r))
    if rewards:
        import statistics as _st
        print(
            f"[Group DONE] rollout={state.current_rollout_id} n={len(group)} k={k} "
            f"rewards: mean={_st.mean(rewards):.4f} min={min(rewards):.4f} max={max(rewards):.4f} "
            f"positive={sum(1 for x in rewards if x > 0)}/{len(rewards)}",
            flush=True,
        )

    # ── Write structured debug trace at first rollout and checkpoint intervals ──
    _save_dir = getattr(args, "save", None)
    _save_interval = getattr(args, "save_interval", None) or 5
    _rid = state.current_rollout_id
    _write_trace = (
        _save_dir
        and (_rid == 0 or (_save_interval and (_rid + 1) % _save_interval == 0))
    )
    if _write_trace:
        _write_rollout_trace(args, group, _rid, k)

    return group


def _write_rollout_trace(
    args: Namespace,
    group: list[Sample],
    rollout_id: int,
    k: int,
) -> None:
    """Write a structured debug trace file for one rollout to the checkpoint dir.

    Contains per-sample, per-step info: parsed code length, snippet preview,
    error type, score.  Enables quick post-hoc diagnosis of parsing bugs,
    prompt quality, etc.
    """
    import json, os
    save_dir = getattr(args, "save", None)
    if not save_dir:
        return
    trace_dir = os.path.join(save_dir, "rollout", "traces")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"trace_rollout_{rollout_id:04d}.jsonl")

    n_total = 0
    n_valid = 0
    n_snippet = 0  # parsed code < 500 chars
    with open(trace_path, "w") as f:
        for s in group:
            traces = (s.metadata or {}).get("_step_traces", [])
            for tr in traces:
                f.write(json.dumps(tr, ensure_ascii=False) + "\n")
                n_total += 1
                if tr.get("valid"):
                    n_valid += 1
                if tr.get("parsed_code_len", 9999) < 500:
                    n_snippet += 1

    print(
        f"[Trace] Wrote {trace_path}: {n_total} step-records, "
        f"valid={n_valid}/{n_total} snippet(<500)={n_snippet}/{n_total}",
        flush=True,
    )


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True
    if args.use_slime_router or parse(sglang_router.__version__) <= parse("0.2.1"):
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    # abort all the requests
    for url in urls:
        print(f"Abort request for {url}", flush=True)
        await post(f"{url}/abort_request", {"abort_all": True})

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        print(f"Collected {count} partial samples into the data buffer", flush=True)

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]], data_buffer=None
) -> tuple[list[list[Sample]], list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[list[list[Sample]], list[list[Sample]]]:
            - completed samples grouped per prompt
            - aborted samples collected during partial rollout
    """
    assert (
        (args.rollout_global_dataset and not args.evolving_gym)
        or (not args.rollout_global_dataset and args.evolving_gym)
    ), "Exactly one of global_dataset / evolving_gym must be enabled."


    state = GenerateState(args)
    state.current_rollout_id = rollout_id          # ← curriculum K reads this

    effective_k = get_effective_k(args, rollout_id)
    x_advance = get_rollout_x_advance(args, rollout_id)
    inner_cum = get_inner_step_cum(args, rollout_id)

    # Detect K-schedule transition (compare with previous rollout)
    prev_k = get_effective_k(args, rollout_id - 1) if rollout_id > 0 else effective_k
    if prev_k != effective_k:
        print(
            f"\n{'🔀'*30}\n"
            f"[K-TRANSITION] rollout={rollout_id}: K changed {prev_k} → {effective_k}  "
            f"inner_cum={inner_cum}\n"
            f"{'🔀'*30}",
            flush=True,
        )

    _log_schedule_info(args, rollout_id)
    print(
        f"[Rollout axis] rollout={rollout_id} k={effective_k} x_advance=+{x_advance} "
        f"inner_cum={inner_cum}",
        flush=True,
    )

    # Wire improvement-rate eviction gym reference so the dynamic sampling filter can classify parents.
    _needs_classify = getattr(args, "evolving_gym_beat_rate_eviction", False)
    if (_needs_classify
            and data_buffer is not None
            and hasattr(data_buffer, "evolving_gym_manager")):
        args._acds_gym = data_buffer.evolving_gym_manager.gym
        try:
            args._acds_gym.database.acds_begin_step()
        except Exception as e:
            print(f"[IR-evict] acds_begin_step error: {e}")

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    # Generate exactly rollout_batch_size groups per step.
    target_data_size = args.rollout_batch_size
    n_total_generated = 0  # total groups generated

    data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()
            n_total_generated += 1

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                print(
                    f"First rollout sample: {[sample.prompt + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                    flush=True,
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            # improvement-rate eviction: classify_parent (mastery detection) happens inside the filter
            if dynamic_filter is not None:
                dynamic_filter(args, group)

            data.append(group)
            pbar.update(args.n_samples_per_prompt)

    pbar.close()
    print(f"[Rollout] Collected {len(data)} groups from {n_total_generated} total generated",
          flush=True)
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    print(
        f"Finish rollout: {[sample.prompt + sample.response]}, label: {sample.label}, reward: {sample.reward}",
        flush=True,
    )

    # Abort any still-pending requests
    _openai_mode_abort = getattr(args, "openai_api_base", None) is not None
    if _openai_mode_abort:
        aborted_samples = []          # no sglang router to contact
    else:
        aborted_samples = await abort(args, rollout_id)
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)

    # === NEW: evolving-gym database update and recording ===
    if getattr(args, "evolving_gym", False):
        assert args.rm_type == "evolving-gym", f"evolving_gym is only supported for evolving-gym, got {args.rm_type}"
        assert data_buffer is not None, "data_buffer must be provided for evolving_gym"
        assert hasattr(data_buffer, "evolving_gym_manager"), "data_source must have evolving_gym_manager attribute"
        gym = data_buffer.evolving_gym_manager.gym
        assert gym is not None, "gym must be provided for evolving_gym"

        # Update database from temp cache (always do this for evolving_gym)
        gym.database.update_database_from_temp(verbose=True)
        print(f"[SUCCESS] Rollout {rollout_id} database update")

        # Island rotation at rollout boundary (after acds_end_step below).

        # Recording (optional, independent of database update)
        if getattr(args, "evolving_gym_record", False) and gym.recording_enabled:
            try:
                # Summarize current round reward and success rate (>0 considered success), using args.reward_key
                reward_key = getattr(args, "reward_key", "reward")
                flat = []
                print("start to flatten the data for reward calculation")
                for group in data:
                    for s in group:
                        r = s.reward
                        if isinstance(r, dict):
                            r = r.get(reward_key, 0.0)
                        if not isinstance(r, (int, float)):
                            r = 0.0
                        flat.append(float(r))
                print(f"flattened data length: {len(flat)}")
                total = len(flat)
                avg_reward = (sum(flat) / total) if total > 0 else 0.0
                success_rate = (sum(1 for x in flat if x > 0.0) / total) if total > 0 else 0.0

                # Database size (convenient for tracking evolution progress)
                db_size = len(gym.database.programs)

                # --- Compute best score and invalid program rate ---
                # Error codes are now tracked via multistep_error_codes in sample.reward
                best_program = gym.database.get_best_program()
                best_score = None
                if best_program and best_program.metrics:
                    best_score = best_program.metrics.get("combined_score", None)

                # Count invalid programs from this batch using explicit validity flags + error codes
                # ⚠️  HARDCODED error-code thresholds — these magic numbers (-0.4 … -0.1)
                #     are defined in evolving_gym_rm.py (ErrorThresholds).  If those
                #     values change, update _classify_error_code() below accordingly.
                # 4 error categories (from error_codes):
                #   -0.4  NO_CODE_BLOCK: LLM didn't output a code block
                #   -0.3  NO_VALID_CHANGE: output identical to parent (lazy)
                #   -0.2  EXECUTION_ERROR: runtime crash / timeout / no solution
                #   -0.1  VALIDATION_FAILED: ran but constraints violated (overlap/boundary)
                n_invalid = 0
                n_no_code_block = 0   # -0.4
                n_no_valid_change = 0  # -0.3
                n_execution_error = 0  # -0.2
                n_validation_failed = 0  # -0.1
                n_total_steps = 0  # total refinement steps across all samples

                def _classify_error_code(ec):
                    """Classify an error code into specific categories."""
                    nonlocal n_no_code_block, n_no_valid_change, n_execution_error, n_validation_failed
                    if ec is None:
                        return
                    if abs(ec - (-0.4)) < 1e-6:
                        n_no_code_block += 1
                    elif abs(ec - (-0.3)) < 1e-6:
                        n_no_valid_change += 1
                    elif abs(ec - (-0.2)) < 1e-6:
                        n_execution_error += 1
                    elif abs(ec - (-0.1)) < 1e-6:
                        n_validation_failed += 1

                for group in data:
                    for s in group:
                        r = s.reward
                        if isinstance(r, dict):
                            mv = r.get("multistep_validities")
                            mec = r.get("multistep_error_codes")
                            if mv and isinstance(mv, (list, tuple)) and len(mv) >= 2:
                                # Multistep mode: use explicit validity + error_codes
                                for t in range(1, len(mv)):
                                    n_total_steps += 1
                                    if not mv[t]:
                                        n_invalid += 1
                                        if mec and t < len(mec):
                                            _classify_error_code(mec[t])
                            else:
                                # Single-step fallback
                                n_total_steps += 1
                                child_metrics = r.get("metrics", {})
                                v = child_metrics.get("validity", None)
                                cs = child_metrics.get("combined_score", None)
                                if v is not None:
                                    is_valid = float(v) > 0.5
                                else:
                                    is_valid = cs is not None and cs >= 0
                                if not is_valid:
                                    n_invalid += 1
                                    _classify_error_code(cs)
                invalid_rate = (n_invalid / n_total_steps) if n_total_steps > 0 else 0.0

                actual_n_groups = len(data)
                step_metrics = {
                    "rollout_id": rollout_id,
                    "batch_operations": actual_n_groups * args.n_samples_per_prompt,
                    "avg_reward": avg_reward,
                    "success_rate": success_rate,
                    "database_size": db_size,
                    "best_score": best_score,
                    "invalid_rate": invalid_rate,
                }
                print(f"[EvolvingGym Recorder] Rollout {rollout_id} metrics: {step_metrics}")

                # --- Compute per-rollout parent/sample stats ---
                n_parents = actual_n_groups  # actual number of trainable groups
                n_children_per_parent = args.n_samples_per_prompt
                n_total_samples = n_parents * n_children_per_parent

                # Current k value (from schedule)
                current_k = getattr(args, "multistep_refine_k", 1)
                k_schedule_str = getattr(args, "multistep_k_schedule", "")
                if k_schedule_str:
                    # Parse schedule to find current k
                    try:
                        phases = []
                        for part in k_schedule_str.split(","):
                            kv, nv = part.split(":")
                            phases.append((int(kv), int(nv)))
                        remaining = rollout_id
                        for kv, nv in phases:
                            if remaining < nv:
                                current_k = kv
                                break
                            remaining -= nv
                        else:
                            current_k = phases[-1][0] if phases else current_k
                    except Exception:
                        pass

                # Collect unique parent scores from this batch
                parent_scores = []
                for group in data:
                    for s in group:
                        r = s.reward
                        if isinstance(r, dict):
                            mrs = r.get("multistep_raw_scores")
                            if mrs and len(mrs) >= 1:
                                parent_scores.append(float(mrs[0]))
                                break  # one per group
                mean_parent_score = (sum(parent_scores) / len(parent_scores)) if parent_scores else 0.0

                # Current rollout batch improvement-rate (group-level).
                # NOTE: improvement-rate eviction parent classification now happens inside the dynamic
                # sampling filter (check_evolving_gym_acds_trainable) so that ALL
                # generated groups — including rejected ones — get classified.
                # Here we only compute batch stats from the accepted (trainable) groups.
                _reward_agg = getattr(args, "multistep_reward_agg", "sum")
                batch_beat_total = 0
                batch_beat_hits = 0
                for group in data:
                    for s in group:
                        r = s.reward
                        if not isinstance(r, dict):
                            continue
                        mrs = r.get("multistep_raw_scores")
                        mvs = r.get("multistep_validities")
                        if not (mrs and len(mrs) >= 2):
                            continue
                        parent_score = float(mrs[0])
                        if _reward_agg in ("max", "max_improve"):
                            child_score = max(float(x) for x in mrs[1:])
                        else:
                            child_score = float(mrs[-1])
                        child_valid = True
                        if isinstance(mvs, list) and len(mvs) >= 2:
                            child_valid = any(mvs[1:])
                        if child_valid and parent_score > 0:
                            batch_beat_total += 1
                            if child_score > parent_score:
                                batch_beat_hits += 1

                # improvement-rate eviction: increment step counter + trigger buffer probe if needed
                # (classification already done in the dynamic sampling filter)
                if gym.database.adaptive_sampling_enabled:
                    try:
                        gym.database.acds_end_step()
                    except Exception as e:
                        print(f"[IR-evict] acds_end_step error: {e}")

                # Island rotation after acds_end_step.
                try:
                    gym.database.advance_island_after_rollout()
                except Exception as e:
                    print(f"[Island] advance error: {e}")

                batch_beat_rate = (
                    (batch_beat_hits / batch_beat_total) if batch_beat_total > 0 else None
                )

                # Context horizon & reward agg
                context_horizon = getattr(args, "context_horizon", 0)
                reward_agg = getattr(args, "multistep_reward_agg", "sum")
                adaptive_k_patience = int(getattr(args, "adaptive_k_patience", 0))

                # --- Adaptive-k per-rollout stats (computed from sample reward dicts) ---
                _adaptive_early_stopped = 0
                _adaptive_total_chains = 0
                _adaptive_steps_saved = 0
                _adaptive_steps_possible = 0
                if adaptive_k_patience > 0 and current_k > 0:
                    for group in data:
                        for s in group:
                            r = s.reward
                            if isinstance(r, dict):
                                ak = r.get("chain_actual_k", None)
                                if isinstance(ak, int) and ak > 0:
                                    _adaptive_total_chains += 1
                                    if ak < current_k:
                                        _adaptive_early_stopped += 1
                                    _adaptive_steps_saved += max(0, current_k - ak)
                                    _adaptive_steps_possible += current_k

                # --- Log to wandb ---
                if getattr(args, "use_wandb", False):
                    global _cum_actual_evals_wandb
                    _cum_actual_evals_wandb += n_total_steps
                    wandb_dict = {
                        "evolving_gym/avg_reward": avg_reward,
                        "evolving_gym/success_rate": success_rate,
                        "evolving_gym/database_size": db_size,
                        "evolving_gym/invalid_rate": invalid_rate,
                        "evolving_gym/n_invalid": n_invalid,
                        "evolving_gym/n_total_steps": n_total_steps,
                        "evolving_gym/n_no_code_block": n_no_code_block,
                        "evolving_gym/n_no_valid_change": n_no_valid_change,
                        "evolving_gym/n_execution_error": n_execution_error,
                        "evolving_gym/n_validation_failed": n_validation_failed,
                        "evolving_gym/step": rollout_id,
                        "evolving_gym/cum_actual_evals": _cum_actual_evals_wandb,
                        # Sampling & config info
                        "evolving_gym/n_parents": n_parents,
                        "evolving_gym/n_children_per_parent": n_children_per_parent,
                        "evolving_gym/n_total_samples": n_total_samples,
                        "evolving_gym/current_k": current_k,
                        "evolving_gym/mean_parent_score": mean_parent_score,
                        "evolving_gym/context_horizon": context_horizon,
                        "evolving_gym/reward_agg": 0 if reward_agg == "sum" else 1,
                        # Adaptive-k config & stats
                        "evolving_gym/adaptive_k_patience": adaptive_k_patience,
                        # Variable batch tracking
                        "evolving_gym/actual_train_groups": actual_n_groups,
                        "evolving_gym/total_generated_groups": n_total_generated,
                    }
                    if adaptive_k_patience > 0 and _adaptive_total_chains > 0:
                        wandb_dict["evolving_gym/adaptive_k_early_stop_count"] = _adaptive_early_stopped
                        wandb_dict["evolving_gym/adaptive_k_early_stop_frac"] = _adaptive_early_stopped / _adaptive_total_chains
                        wandb_dict["evolving_gym/adaptive_k_steps_saved"] = _adaptive_steps_saved
                        wandb_dict["evolving_gym/adaptive_k_saved_frac"] = (
                            _adaptive_steps_saved / _adaptive_steps_possible if _adaptive_steps_possible > 0 else 0.0
                        )
                    if n_total_steps > 0:
                        wandb_dict["evolving_gym/no_code_block_rate"] = n_no_code_block / n_total_steps
                        wandb_dict["evolving_gym/no_valid_change_rate"] = n_no_valid_change / n_total_steps
                        wandb_dict["evolving_gym/execution_error_rate"] = n_execution_error / n_total_steps
                        wandb_dict["evolving_gym/validation_failed_rate"] = n_validation_failed / n_total_steps
                    if best_score is not None:
                        wandb_dict["evolving_gym/best_score"] = best_score
                    if best_program and hasattr(best_program, "code") and best_program.code:
                        import html as _html_mod
                        _escaped = _html_mod.escape(best_program.code)
                        wandb_dict["evolving_gym/best_program"] = wandb.Html(
                            f"<pre style='white-space:pre-wrap;font-size:12px'>{_escaped}</pre>"
                        )
                    # ── improvement-rate eviction metrics ──
                    try:
                        if gym.database.adaptive_sampling_enabled:
                            acds = gym.database.get_acds_stats()
                            wandb_dict["evolving_gym/db_size"] = acds["pool_size"]
                            wandb_dict["evolving_gym/acds_avg_beat_rate"] = acds["avg_beat_rate"]
                            wandb_dict["evolving_gym/sweep_threshold"] = acds.get("sweep_threshold", 0.0)
                            wandb_dict["evolving_gym/retire_thr"] = acds.get("current_retire_thr", 0.0)
                            wandb_dict["evolving_gym/evict_count_sweep"] = acds.get("sweep_size", 0)
                            wandb_dict["evolving_gym/mastered_count"] = acds.get("mastered_count", 0)
                            # All-programs mean score
                            all_scores = [
                                p.metrics.get("combined_score", 0.0)
                                for p in gym.database.programs.values()
                                if isinstance(p.metrics.get("combined_score"), (int, float))
                            ]
                            if all_scores:
                                wandb_dict["evolving_gym/db_mean_score"] = sum(all_scores) / len(all_scores)
                        if batch_beat_rate is not None:
                            wandb_dict["evolving_gym/beat_rate_batch"] = batch_beat_rate
                    except Exception:
                        pass
                    try:
                        wandb.log(wandb_dict)
                        print(f"[EvolvingGym wandb] Logged: best_score={best_score}, invalid_rate={invalid_rate:.3f}, "
                              f"n_parents={n_parents}, n_children={n_children_per_parent}, k={current_k}, "
                              f"parent_score={mean_parent_score:.4f}, ctx_horizon={context_horizon}, "
                              f"cum_actual_evals={_cum_actual_evals_wandb}, "
                              f"B_batch={(batch_beat_rate if batch_beat_rate is not None else -1):.3f}")
                    except Exception as e:
                        print(f"[EvolvingGym wandb] Failed to log: {e}")

                # Record data for extracted_code and runtime_environment, but skip redundant visualization
                # (use wandb for real-time monitoring instead)
                
                print(f"start to record progress for rollout {rollout_id}")
                gym.record_progress(
                    training_step=rollout_id,
                    step_metrics=step_metrics,
                    save_checkpoint=False,
                    save_historical_records=False,
                )

                print(f"finish recording progress for rollout {rollout_id}")
                if gym.recording_enabled and gym._recorder:
                    gym._recorder.print_database_score_distribution()

                    gym.database.log_island_status()
                else:
                    print(f"[DEBUG] skipping gym.record_progress since recording_enabled is False")

                # ── improvement-rate eviction status ──
                if gym.database.adaptive_sampling_enabled:
                    acds = gym.database.get_acds_stats()

                    # Improvement-rate histogram for this rollout
                    _all_bh = list(gym.database.acds_beat_history)
                    _cur_retire = acds.get('current_retire_thr', 0.0)
                    _n_bh = len(_all_bh)
                    # Bucket improvement rates
                    _br_vals = [b for _, b in _all_bh[-100:]]  # last ~100 (one rollout)
                    _hist = {}
                    for _b in _br_vals:
                        _bucket = round(_b, 3)
                        _hist[_bucket] = _hist.get(_bucket, 0) + 1
                    _n_above = sum(1 for _b in _br_vals if _b >= _cur_retire - 1e-9)
                    _n_br = len(_br_vals)
                    # Mean mastered score by improvement-rate tier
                    _tier_scores = {}
                    for _s, _b in _all_bh[-100:]:
                        if _b >= _cur_retire - 1e-9:
                            _bucket = round(_b, 3)
                            _tier_scores.setdefault(_bucket, []).append(_s)
                    _tier_summary = {k: (len(v), round(sum(v)/len(v), 4))
                                     for k, v in sorted(_tier_scores.items())}

                    print(f"[IR-evict] rollout={rollout_id}  "
                          f"db_size={acds['pool_size']}  "
                          f"sweep_thr={acds.get('sweep_threshold', 0.0):.4f}  "
                          f"retire_thr={_cur_retire:.3f}  "
                          f"mastered_n={acds.get('mastered_count', 0)}  "
                          f"evict_sweep={acds.get('sweep_size', 0)}  "
                          f"B_avg={acds['avg_beat_rate']:.3f}  "
                          f"above_thr={_n_above}/{_n_br}({_n_above*100//_n_br if _n_br else 0}%)  "
                          f"hist={dict(sorted(_hist.items()))}  "
                          f"tier_mean_scores={_tier_summary}")

            except Exception as e:
                # Recording failure doesn't affect training
                import traceback
                print(f"[EvolvingGym Recorder] record failed at rollout {rollout_id}: {e}", flush=True)
                print(f"[EvolvingGym Recorder] Full traceback: {traceback.format_exc()}", flush=True)

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    return data, aborted_samples



EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    results = {}
    for i in range(0, len(args.eval_prompt_data), 2):
        name, path = args.eval_prompt_data[i : i + 2]
        results.update(await eval_rollout_single_dataset(args, rollout_id, name, path))
    return results, []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, name: str, path: str
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        name: str, the name of the dataset
        path: str, the path of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    if name not in EVAL_PROMPT_DATASET:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[name] = Dataset(
            path,
            tokenizer=tokenizer,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key if args.eval_input_key is None else args.eval_input_key,
            label_key=args.label_key if args.eval_label_key is None else args.eval_label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key if args.eval_tool_key is None else args.eval_tool_key,
            apply_chat_template=args.apply_chat_template,
            enable_thinking=getattr(args, "enable_thinking", None),
        )
    dataset = EVAL_PROMPT_DATASET[name]

    sampling_params = dict(
        temperature=args.rollout_temperature if args.eval_temperature is None else args.eval_temperature,
        top_p=args.rollout_top_p if args.eval_top_p is None else args.eval_top_p,
        top_k=args.rollout_top_k if args.eval_top_k is None else args.eval_top_k,
        max_new_tokens=(
            args.rollout_max_response_len if args.eval_max_response_len is None else args.eval_max_response_len
        ),
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for i, prompt_sample in enumerate(dataset.samples):
        for j in range(args.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                generate_and_rm(
                    args,
                    sample,
                    sampling_params=sampling_params,
                    evaluation=True,
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc="Rollout generation", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            print([sample.prompt + sample.response], sample.reward, flush=True)
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
        }
    }


# TODO remove this temp function
def generate_rollout(
    args: Namespace, rollout_id: int, data_buffer: Any, evaluation: bool = False
) -> list[list[Sample]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    completed_samples, aborted_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, evaluation=evaluation,
        data_buffer=data_buffer
    )
    data_buffer.add_samples(aborted_samples)
    return completed_samples


def generate_abortable_samples(
    args: Namespace,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
    evaluation: bool = False,
    data_buffer=None,
) -> tuple[Any, list[list[Sample]]]:
    # assert args.rollout_global_dataset
    assert (
        (args.rollout_global_dataset and not args.evolving_gym)
        or (not args.rollout_global_dataset and args.evolving_gym)
    ), "Exactly one of global_dataset / evolving_gym must be enabled."
    if evaluation:
        return run(eval_rollout(args, rollout_id))
    return run(generate_rollout_async(args, rollout_id, data_source, data_buffer=data_buffer))
