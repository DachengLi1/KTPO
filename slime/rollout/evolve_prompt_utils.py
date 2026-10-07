"""
Shared prompt rewriting and feedback utilities for the evolutionary code loop.

This module is intentionally dependency-free (only stdlib `re`) so it can be
imported by BOTH:
  - slime/rollout/sglang_rollout.py  (training path, inside Ray/sglang)
  - infer_evolve.py                  (inference-only path, uses external vllm)

Any change here affects both paths — single source of truth.
"""

import re
from typing import Optional

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Constants & instruction templates                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Language-aware instruction templates.
# `language` should be "python" or "cpp".

def _get_code_fence_lang(language: str) -> str:
    """Return the code fence language tag."""
    return "cpp" if language in ("cpp", "c++", "c") else "python"


def _get_full_program_instructions(language: str = "python") -> str:
    lang_tag = _get_code_fence_lang(language)
    if lang_tag == "cpp":
        return """\
- Output Format -

Output the COMPLETE, SELF-CONTAINED C++ program as a single ```cpp code block.
Your code block MUST include ALL #include headers, ALL helper functions/structs, and a complete `int main()` function that reads from stdin and writes to stdout.
Do NOT output partial snippets, diffs, or incomplete code — every code block you produce will be compiled and executed as a standalone program.
IMPORTANT: Your program must have `int main()` that handles I/O correctly. Do NOT define a separate entry point function — use main() directly.
The reference programs shown above are for inspiration only — you are free to try a completely different approach.

- Reward Rules -

You will get multiple refinement attempts. After each attempt, you will see your score and can try again.
- Your reward is based on the BEST score you achieve across all attempts minus your starting score.
- You get PENALIZED for producing INVALID programs (compile errors, runtime errors, or wrong output format).
- You get PENALIZED for producing the SAME score as your previous attempt (lazy repetition).
Therefore: always aim to improve, avoid generating broken code, and do not repeat the same solution.

- Task -

Generate the C++ program code that maximizes the evaluation score.
Output your code as a single ```cpp code block — complete and compilable.
Be creative and try different approaches. Explain your reasoning briefly before the code."""
    else:
        return """\
- Output Format -

Output the COMPLETE, SELF-CONTAINED program as a single ```python code block.
Your code block MUST include ALL imports, ALL helper functions, and the main entry-point function (keep the same function name as the reference program).
Do NOT output partial snippets, diffs, or incomplete code — every code block you produce will be executed as a standalone program.
IMPORTANT: Keep the same entry-point function name as in the reference program — this is the function that will be called to evaluate your solution. Do not rename it.
The reference programs shown above are for inspiration only — you are free to try a completely different approach.

- Reward Rules -

You will get multiple refinement attempts. After each attempt, you will see your score and can try again.
- Your reward is based on the BEST score you achieve across all attempts minus your starting score.
- You get PENALIZED for producing INVALID programs (syntax errors, runtime errors, or constraint violations).
- You get PENALIZED for producing the SAME score as your previous attempt (lazy repetition).
Therefore: always aim to improve, avoid generating broken code, and do not repeat the same solution.

- Task -

Generate the program code that maximizes the evaluation score.
Output your code as a single ```python code block — complete and runnable.
Be creative and try different approaches. Explain your reasoning briefly before the code."""


def _get_single_ref_instructions(language: str = "python") -> str:
    lang_tag = _get_code_fence_lang(language)
    if lang_tag == "cpp":
        return """\
- Output Format -

Output the COMPLETE, SELF-CONTAINED C++ program as a single ```cpp code block.
Your code block MUST include ALL #include headers, ALL helper functions/structs, and a complete `int main()` function that reads from stdin and writes to stdout.
Do NOT output partial snippets, diffs, or incomplete code — every code block you produce will be compiled and executed as a standalone program.
IMPORTANT: Your program must have `int main()` that handles I/O correctly. Do NOT define a separate entry point function — use main() directly.
Your job is to IMPROVE the reference program shown above — build on it, fix its weaknesses, and push the score higher.
Do NOT start from scratch or discard the reference; make targeted improvements to it.

- Reward Rules -

You will get multiple refinement attempts. After each attempt, you will see your score and can try again.
- Your reward is based on the BEST score you achieve across all attempts minus your starting score.
- You get PENALIZED for producing INVALID programs (compile errors, runtime errors, or wrong output format).
- You get PENALIZED for producing the SAME score as your previous attempt (lazy repetition).
Therefore: always aim to improve, avoid generating broken code, and do not repeat the same solution.

- Task -

Improve the reference program above and output the complete revised C++ program.
Output your code as a single ```cpp code block — complete and compilable.
Explain what specific changes you made and why before the code."""
    else:
        return """\
- Output Format -

Output the COMPLETE, SELF-CONTAINED program as a single ```python code block.
Your code block MUST include ALL imports, ALL helper functions, and the main entry-point function (keep the same function name as the reference program).
Do NOT output partial snippets, diffs, or incomplete code — every code block you produce will be executed as a standalone program.
IMPORTANT: Keep the same entry-point function name as in the reference program — this is the function that will be called to evaluate your solution. Do not rename it.
Your job is to IMPROVE the reference program shown above — build on it, fix its weaknesses, and push the score higher.
Do NOT start from scratch or discard the reference; make targeted improvements to it.

- Reward Rules -

You will get multiple refinement attempts. After each attempt, you will see your score and can try again.
- Your reward is based on the BEST score you achieve across all attempts minus your starting score.
- You get PENALIZED for producing INVALID programs (syntax errors, runtime errors, or constraint violations).
- You get PENALIZED for producing the SAME score as your previous attempt (lazy repetition).
Therefore: always aim to improve, avoid generating broken code, and do not repeat the same solution.

- Task -

Improve the reference program above and output the complete revised program.
Output your code as a single ```python code block — complete and runnable.
Explain what specific changes you made and why before the code."""


# Backward-compat aliases (used if someone imports the old constants directly)
FULL_PROGRAM_INSTRUCTIONS = _get_full_program_instructions("python")
SINGLE_REF_INSTRUCTIONS = _get_single_ref_instructions("python")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Traceback / boilerplate stripping                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

def strip_traceback_boilerplate(error_text: str) -> str:
    """Strip framework boilerplate frames from Python tracebacks.

    The evaluator wraps the model's code with boilerplate (run_*,
    save_search_results, matplotlib, etc.).  When errors occur, the traceback
    includes these framework frames.  We strip them so the model only sees
    frames from the user-written code – the code it actually wrote.

    Also strips stdout output that leaks boilerplate info.
    """
    if not error_text:
        return error_text

    # Framework boilerplate keywords — applies to all tasks
    _BOILERPLATE_KEYWORDS = (
        "run_circle_packing",
        "run_erdos_optimization",
        "save_search_results",
        "save_checkpoint",
        "file_io_controller",
        "evaluation_controller",
        "get_current_problem_config",
        "matplotlib",
        "plt.",
    )

    lines = error_text.split("\n")
    cleaned: list[str] = []
    skip_next = False
    for line in lines:
        if skip_next:
            skip_next = False
            continue
        if any(kw in line for kw in _BOILERPLATE_KEYWORDS):
            stripped = line.lstrip()
            if stripped.startswith("File "):
                skip_next = True
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def strip_boilerplate_from_code_blocks(prompt_text: str, language: str = "python") -> str:
    """Strip boilerplate outside EVOLVE-BLOCK markers from code blocks.

    In the prompt, each code block may contain the full program file
    (EVOLVE-BLOCK + main/boilerplate). We keep only the EVOLVE-BLOCK content
    so the model sees only the editable section (SkyAgent style).
    
    Works for both Python and C++ — extracts only the EVOLVE-BLOCK content
    to keep prompts short.
    """
    lang_tag = _get_code_fence_lang(language)

    if lang_tag == "cpp":
        # For C++: strip boilerplate outside EVOLVE-BLOCK, keeping only the
        # EVOLVE-BLOCK content.  After EVOLVE-BLOCK replacement the model's
        # full program lives INSIDE the block, so this is safe and drastically
        # reduces prompt token count (avoids duplicated headers in references).
        _cpp_fence_re = r"```(?:cpp|c\+\+)\n(.*?)```"
        _cpp_eb_re = (
            r"^[ \t]*(?://|/\*)?\s*EVOLVE-BLOCK-START\s*\n"
            r"(.*?)\n"
            r"[ \t]*(?://|/\*)?\s*EVOLVE-BLOCK-END\b"
        )

        def _replace_cpp_block(m: re.Match) -> str:
            full_code = m.group(1)
            eb = re.search(_cpp_eb_re, full_code, re.DOTALL | re.MULTILINE)
            if eb:
                return f"```cpp\n{eb.group(1).rstrip()}\n```"
            return m.group(0)  # no EVOLVE-BLOCK found → keep as-is

        return re.sub(_cpp_fence_re, _replace_cpp_block, prompt_text, flags=re.DOTALL)

    # Python path: original behavior
    def _replace_block(m: re.Match) -> str:
        full_code = m.group(1)
        eb = re.search(
            r"^[ \t]*#?\s*EVOLVE-BLOCK-START\s*\n(.*?)\n[ \t]*#?\s*EVOLVE-BLOCK-END\b",
            full_code,
            re.DOTALL | re.MULTILINE,
        )
        if eb:
            return f"```python\n{eb.group(1).rstrip()}\n```"
        return m.group(0)

    return re.sub(r"```python\n(.*?)```", _replace_block, prompt_text, flags=re.DOTALL)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Prompt rewriting: diff-mode → full-program mode                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def rewrite_prompt_for_full_program(prompt_text: str, single_reference: bool = False, language: str = "python") -> str:
    """Rewrite the diff-mode prompt into full-program mode.

    Args:
        prompt_text: The original prompt text (diff-style from OpenEvolve).
        single_reference: If True, strip "Prior programs" entirely and keep only
            the current program as the single reference the model must improve.
        language: Programming language — "python" or "cpp".

    Steps (both modes):
        1. Handle reference programs section (merge or strip depending on mode).
        2. Replace SEARCH/REPLACE instructions with full-program instructions.
        3. Strip boilerplate from code blocks (keep only EVOLVE-BLOCK content).
        4. Deduplicate identical Reference Program blocks.
    """
    lang_tag = _get_code_fence_lang(language)

    if single_reference:
        prompt_text = re.sub(
            r"- Prior programs -\n.*?(?=- Current program -)",
            "",
            prompt_text,
            count=1,
            flags=re.DOTALL,
        )
        prompt_text = re.sub(
            r"- Current program -\n.*?(?=-{5,}\s*Current Program\s*-{5,})",
            "- Reference program (your job is to IMPROVE this program) -\n"
            "Here is the program you must improve. Build on it — do not discard it.\n\n",
            prompt_text,
            count=1,
            flags=re.DOTALL,
        )
        prompt_text = re.sub(
            r"-{5,}\s*Current Program\s*-{5,}",
            "---------------- Reference Program ----------------",
            prompt_text,
        )
        instructions = _get_single_ref_instructions(language)
    else:
        prompt_text = re.sub(
            r"- Prior programs -\n.*?(?=\n-{5,})",
            "- Reference programs -\n"
            "The following are known good solutions for this task — treat them as a starting point or source of inspiration. "
            "You may refine one of them, combine ideas from multiple, "
            "or write a completely different approach.\n",
            prompt_text,
            count=1,
            flags=re.DOTALL,
        )
        prompt_text = re.sub(r"Prior Program (\d+)", r"Reference Program \1", prompt_text)

        ref_nums = [int(m) for m in re.findall(r"Reference Program (\d+)", prompt_text)]
        next_num = max(ref_nums) + 1 if ref_nums else 1

        prompt_text = re.sub(
            r"- Current program -\n.*?-{5,}\s*Current Program\s*-{5,}",
            f"---------------- Reference Program {next_num} ----------------",
            prompt_text,
            count=1,
            flags=re.DOTALL,
        )
        instructions = _get_full_program_instructions(language)

    # ── Step 2: Replace diff instructions with full-program instructions ──
    start_marker = "- SEARCH/REPLACE block rules -"
    end_marker = "IMPORTANT: Do not rewrite the entire program"

    start_idx = prompt_text.find(start_marker)
    if start_idx != -1:
        end_idx = prompt_text.find(end_marker, start_idx)
        if end_idx == -1:
            end_idx = start_idx + len(start_marker)
        else:
            end_idx = prompt_text.find("\n", end_idx)
            if end_idx == -1:
                end_idx = len(prompt_text)
            else:
                end_idx += 1
        prompt_text = prompt_text[:start_idx] + instructions + "\n" + prompt_text[end_idx:]

    # ── Step 3: Strip boilerplate from code blocks ──
    prompt_text = strip_boilerplate_from_code_blocks(prompt_text, language=language)

    # ── Step 4: Deduplicate identical Reference Program blocks ──
    # Match code blocks in any language fence (python, cpp, c++)
    ref_pattern = re.compile(
        r"(-{5,}\s*Reference Program \d+\s*-{5,}\n.*?```(?:python|cpp|c\+\+)\n)(.*?)(```)",
        re.DOTALL,
    )
    seen_codes: dict = {}
    blocks_to_remove: list = []
    for m in ref_pattern.finditer(prompt_text):
        code = m.group(2).strip()
        full_block = m.group(0)
        if code in seen_codes:
            blocks_to_remove.append(full_block)
        else:
            seen_codes[code] = full_block
    for block in blocks_to_remove:
        prompt_text = prompt_text.replace(block, "")
    prompt_text = re.sub(r"\n{3,}", "\n\n", prompt_text)
    counter = [0]
    def _renumber(m: re.Match) -> str:
        counter[0] += 1
        return f"Reference Program {counter[0]}"
    prompt_text = re.sub(r"Reference Program \d+", _renumber, prompt_text)

    return prompt_text


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Multi-step feedback formatting                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def format_multistep_feedback(
    step: int,
    total_steps: int,
    score: float,
    best_score: float,
    metrics: dict,
    error: Optional[str],
    cumulative_attempt: Optional[int] = None,
    chain_best_score: Optional[float] = None,
    target_score: Optional[float] = None,
    db_worst_score: Optional[float] = None,
    no_improve_streak: int = 0,
    adaptive_k_patience: int = 0,
    parent_score: Optional[float] = None,
    step_scores_history: Optional[list] = None,
    language: str = "python",
) -> str:
    """Format a concise feedback message inserted between refinement steps.

    Args:
        cumulative_attempt: If set, use this as a plain attempt counter
            instead of "step X/Y, Z remaining".  Used for cross-rollout
            trajectory history so the model isn't confused by step resets.
        chain_best_score: Best score achieved so far in this chain (including parent).
        target_score: Known target / best-known score for this task.
        db_worst_score: Current worst score in the evolution database (survival threshold).
        no_improve_streak: Consecutive steps that failed to beat db_worst_score.
        adaptive_k_patience: Steps below threshold before chain is terminated.
        parent_score: Score of the reference program shown in the initial prompt.
        step_scores_history: Scores of all previous steps in this chain (excluding current).
    """
    # Task-specific metric keys
    lang_tag = _get_code_fence_lang(language)

    if cumulative_attempt is not None:
        header_tag = f"[Attempt {cumulative_attempt}]"
    else:
        remaining = total_steps - step
        header_tag = f"[Step {step}/{total_steps}, {remaining} remaining]"

    # Score line: this step + domain-specific diagnostics
    score_parts = [f"this_step={score:.4f}"]
    if error:
        # Safety: truncate massive error messages to avoid token explosion
        _err_display = error if len(error) <= 20000 else error[:20000] + f"... [truncated, {len(error)} chars]"
        score_parts.append(f"error={_err_display}")
    if lang_tag == "cpp":
        # C++ tasks: show judge status if available
        judge_status = metrics.get("judge_status")
        if judge_status and judge_status != "done":
            score_parts.append(f"judge_status={judge_status}")
    else:
        # Python tasks: show domain-specific metrics (if present)
        validity = metrics.get("validity")
        boundary_violations = metrics.get("boundary_violations")
        overlaps = metrics.get("overlaps")
        objective_value = metrics.get("objective_value")
        if validity is not None:
            score_parts.append(f"validity={validity}")
        if boundary_violations is not None:
            score_parts.append(f"boundary_violations={boundary_violations}")
        if overlaps is not None:
            score_parts.append(f"overlaps={overlaps}")
        if objective_value is not None:
            score_parts.append(f"objective_value={objective_value}")

    header = header_tag + "  " + "  ".join(score_parts)

    # Score context: reference program + step history
    context_lines = []
    if parent_score is not None:
        context_lines.append(f"Reference program score: {parent_score:.4f}")
    if step_scores_history:
        history_str = "  ".join(f"{s:.4f}" for s in step_scores_history)
        context_lines.append(f"Your scores so far:      {history_str}  → {score:.4f} (this step)")

    threshold_lines: list = []

    lang_tag = _get_code_fence_lang(language)
    if lang_tag == "cpp":
        guidance = (
            "\nThink about how to improve this solution further. "
            "Output your COMPLETE revised C++ program as a single ```cpp code block. "
            "The code block must be fully self-contained "
            "(all #includes, all helper functions/structs, and int main() with correct I/O) — "
            "do NOT output partial snippets or diffs."
        )
    else:
        guidance = (
            "\nThink about how to improve this solution further. "
            "Output your COMPLETE revised program as a single ```python code block. "
            "The code block must be fully self-contained "
            "(all imports, all function definitions, the main entry-point function) — "
            "do NOT output partial snippets or diffs."
        )

    parts = [header]
    if context_lines:
        parts.append("\n" + "\n".join(context_lines))
    if threshold_lines:
        parts.append("\n" + "\n".join(threshold_lines))
    parts.append(guidance)
    return "".join(parts)


def extract_evolve_block(program_code: str) -> str:
    """Extract the content inside EVOLVE-BLOCK markers from a full program.
    
    Supports both Python (#) and C++ (//) comment markers.
    """
    if not program_code:
        return ""
    m = re.search(
        r"^[ \t]*(?:#|//|/\*)?\s*EVOLVE-BLOCK-START\s*\n(.*?)\n[ \t]*(?:#|//|/\*)?\s*EVOLVE-BLOCK-END\b",
        program_code,
        re.DOTALL | re.MULTILINE,
    )
    return m.group(1).rstrip() if m else program_code.strip()

