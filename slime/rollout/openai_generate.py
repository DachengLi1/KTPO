"""
Drop-in replacement for sglang_rollout.generate() using an OpenAI-compatible
chat API (vllm / sglang / any /v1/chat/completions endpoint).

Loaded ONLY when --openai-api-base is set.  Concurrency is controlled by
GenerateState.semaphore (same as sglang path).
"""

from argparse import Namespace
from typing import Any

_CLIENT = None
_TOKENIZER = None


def _init(args: Namespace):
    global _CLIENT, _TOKENIZER
    if _CLIENT is not None:
        return
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    _CLIENT = AsyncOpenAI(
        base_url=args.openai_api_base,
        api_key=getattr(args, "openai_api_key", "EMPTY"),
    )
    _TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    print(f"[OpenAI] client ready: base={args.openai_api_base} model={args.openai_model}")


async def generate_openai(
    args: Namespace,
    sample: "Sample",
    sampling_params: dict[str, Any],
) -> "Sample":
    """Same contract as ``sglang_rollout.generate()``."""
    from slime.utils.types import Sample

    _init(args)
    messages = (sample.metadata or {}).get("_chat_messages")
    if not messages:
        sample.status = Sample.Status.ABORTED
        return sample

    try:
        resp = await _CLIENT.chat.completions.create(
            model=args.openai_model,
            messages=messages,
            temperature=sampling_params.get("temperature", 1.0),
            max_tokens=sampling_params.get("max_new_tokens", 32768),
        )
    except Exception as e:
        print(f"[OpenAI] API error: {e}")
        sample.status = Sample.Status.ABORTED
        return sample

    new_text = resp.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": new_text})  # keep in sync

    # Token-level bookkeeping (for loss_mask / response_length compatibility)
    n = len(_TOKENIZER(new_text, add_special_tokens=False)["input_ids"])
    sample.tokens = (sample.tokens or []) + [0] * n
    sample.response += new_text
    sample.response_length += n
    if sample.rollout_log_probs is not None:
        sample.rollout_log_probs += [0.0] * n

    sample.status = (
        Sample.Status.TRUNCATED
        if resp.choices[0].finish_reason == "length"
        else Sample.Status.COMPLETED
    )
    return sample
