import asyncio
import re
import uuid
import time
import logging
from typing import Any, Dict, Tuple, Optional

from openevolve.config import Config, load_config
from openevolve.database import Program, ProgramDatabase
from openevolve.prompt.sampler import PromptSampler
from openevolve.evaluator import Evaluator
from openevolve.iteration import Result
from openevolve.utils.code_utils import (
    extract_code_language,
    check_code_identical,
    parse_full_rewrite,
)
from openevolve.modular_utils.error_constants import ErrorCodes, ErrorMessages
from openevolve.modular_utils.score_transform import create_score_transform_config_from_dict
from openevolve.utils.metrics_utils import create_evaluation_metrics
from openevolve.utils.performance_utils import timed_async_operation, timed_operation

logger = logging.getLogger(__name__)


class SingleTaskEvolvingGym:
    """
    Evolving gym for single-task RL-style batching.

    Key features
    - Uses asyncio.Semaphore to limit concurrent evaluations;
    - Added concurrent lock and TTL/capacity limits for prompt debug cache to avoid race conditions and memory growth.
    - Evaluation supports timeout (read from config, default 300s).
    - Result return type changed to Optional[Result]
    """

    # Prompt debug cache strategy
    _TEMP_PROMPT_TTL_SEC = 300          # Temporary keys live for 5 minutes
    _PROMPT_CACHE_MAX = 5000            # Max 5000 entries (including temporary + permanent items)

    def __init__(self,
                 initial_program_path: str,
                 evaluation_file: str,
                 config_path: Optional[str] = None,
                 config: Optional[Config] = None,
                 max_concurrent_evaluations: int = 4,
                 log_prompts: bool = True,
                 lazy_output_penalty_level: int = 1,
                 reward_process_type: str = "original_reward",
                 seed: Optional[int] = None,
                 beat_rate_eviction: bool = False,
                 acds_retire_thr: float = 1.0,
                 acds_retire_thr_end: float = 0.1,
                 total_rollouts: int = 200,
                 acds_mastered_window: int = 8):
        """
        Initialize gym (no I/O or async evaluation; trigger initial evaluation via initialize/create)

        Args:
            initial_program_path: Path to initial program file
            evaluation_file: Path to evaluator file
            config_path: Optional config file path
            config: Optional config object
            max_concurrent_evaluations: Maximum concurrent evaluations allowed
            log_prompts: Whether to log prompt/debug info
            lazy_output_penalty_level: Lazy output penalty level (0=no penalty, 1=check parent, 2=check database)
            reward_process_type: Reward processing type
            seed: Random seed for reproducibility (overrides config.random_seed if provided)
            beat_rate_eviction: Enable improvement-rate eviction
            acds_retire_thr: Beat-rate mastery threshold (start value for decay)
            acds_retire_thr_end: End value for linear decay
            total_rollouts: Total rollouts for decay schedule
            acds_mastered_window: Window size for mastered-score deque (sweep threshold)
        """
        # Load configuration
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)

        # Seed parameter takes priority over config.random_seed
        if seed is not None:
            self.config.random_seed = seed
            print(f"[SingleTaskEvolvingGym] Override config random_seed with seed={seed}")

        # Load initial program
        self.initial_program_path = initial_program_path
        with open(initial_program_path, 'r') as f:
            self.initial_program_code = f.read()
        if not self.config.language:
            self.config.language = extract_code_language(self.initial_program_code) or "python"

        # Initialize components
        self.prompt_sampler = PromptSampler(self.config.prompt)

        if self.config.random_seed is not None:
            self.config.database.random_seed = self.config.random_seed
        self.database = ProgramDatabase(self.config.database)

        # Enable improvement-rate eviction
        if beat_rate_eviction:
            self.database.configure_beat_rate_eviction(
                retire_thr=acds_retire_thr,
                retire_thr_end=acds_retire_thr_end,
                total_rollouts=total_rollouts,
                mastered_window=acds_mastered_window,
            )
            print(f"[SingleTaskEvolvingGym] improvement-rate eviction ENABLED "
                  f"(retire_thr={acds_retire_thr}→{acds_retire_thr_end} "
                  f"over {total_rollouts} rollouts, mastered_window={acds_mastered_window})")

        self.evaluator = Evaluator(
            self.config.evaluator,
            evaluation_file,
            database=self.database,
            executor_max_workers=max_concurrent_evaluations,
        )

        # Concurrency control — single pool for ALL subprocess evaluations.
        # Both initial evals and n-repeat evals share the same semaphore,
        # so total concurrent subprocesses never exceeds max_concurrent_evaluations.
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self._eval_pool = asyncio.Semaphore(max_concurrent_evaluations)

        # Lazy output penalty configuration
        self.lazy_output_penalty_level = lazy_output_penalty_level

        # Reward processing configuration
        self.reward_process_type = reward_process_type

        # Prompt debug data
        self.log_prompts = log_prompts
        self._program_prompts: Optional[Dict[str, dict]] = {} if log_prompts else None
        self._prompts_lock = asyncio.Lock()  # Protect concurrent writes to _program_prompts

        output_str = (
            f"Initial program loaded from {initial_program_path}, "
            f"language={self.config.language}, "
            f"max_concurrent_evaluations={max_concurrent_evaluations}, "
            f"log_prompts={log_prompts}, "
            f"lazy_output_penalty_level={lazy_output_penalty_level}, "
            f"random_seed={self.config.random_seed}"
        )
        # logger.info(output_str)
        print(output_str)

        # Initial evaluation state
        self._initial_program_id = str(uuid.uuid4())
        self._initialized = False 

        # Optional recorder
        self._recorder = None

        # Initialize score_transform_config for consistent metrics creation
        self.score_transform_config = create_score_transform_config_from_dict(self.config.to_dict())
        print(f"Initialized score_transform_config: {self.score_transform_config.__dict__ if self.score_transform_config else None}")

    # ---------------------------
    # Initialization paths (initial evaluation runs only once)
    # ---------------------------

    @timed_async_operation("Gym Initialize")
    async def initialize(self) -> None:
        """
        Async initialization: perform one-time evaluation of initial program and add to database.
        """
        if self._initialized:
            return

        logger.info("Evaluating initial program during initialization (async)")
        initial_metrics = await self._evaluate_initial_program_async()

        initial_program = Program(
            id=self._initial_program_id,
            code=self.initial_program_code,
            language=self.config.language,
            metrics=initial_metrics,
            parent_id=None,
            generation=0,
            iteration_found=0
        )
        self.database.add(initial_program, iteration=0)
        # Create initial snapshot for lazy penalty level 2
        self.database.init_old_code_hashes()
        if self._initial_program_id in self.database.programs:
            print(
                f"[InitialProgram] Added to database with ID: {self._initial_program_id}, metrics: {initial_metrics}"
            )
        else:
            print(
                f"[InitialProgram] WARNING: Program {self._initial_program_id} was REJECTED by the pool entry gate! "
                f"Database is empty ({len(self.database.programs)} programs). metrics: {initial_metrics}"
            )
        # assert False, "debug stop here"
        self._initialized = True

    def initialize_sync(self) -> None:
        """
        Synchronous initialization: one-time initial evaluation entry point for non-async environments.
        - If current thread already has a running loop, raises RuntimeError to avoid nested event loops.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run
            asyncio.run(self.initialize())
        else:
            # When event loop already exists, synchronous wait causes nesting/deadlock, explicitly error
            raise RuntimeError(
                "initialize_sync() cannot be called when an event loop is running. "
                "Use `await gym.initialize()` instead."
            )

    async def _evaluate_initial_program_async(self) -> Dict:
        """Async helper for initial program evaluation"""
        try:
            # Evaluation timeout (seconds), default 1000
            timeout_s = getattr(self.config.evaluator, "timeout_s", None) or getattr(self.config.evaluator, "timeout", None) or 1000
            return await asyncio.wait_for(
                self.evaluator.evaluate_program(self.initial_program_code, self._initial_program_id),
                timeout=timeout_s
            )
        except asyncio.TimeoutError:
            logger.warning("Initial program evaluation timed out")
            return {}
        except Exception as e:
            logger.warning(f"Initial program evaluation failed: {e}")
            return {}

    # ---------------------------
    # Prompt recording/association (concurrency-safe)
    # ---------------------------

    async def _associate_prompt_with_program(self, program_id: str, parent_id: str, response: str):
        """Associate temporary prompt data with real program_id (concurrency-safe)"""
        if not self.log_prompts or self._program_prompts is None:
            return

        try:
            async with self._prompts_lock:
                matching_temp_keys = [
                    k for k in self._program_prompts.keys()
                    if k.startswith(f"temp_{parent_id}_")
                ]

                if matching_temp_keys:
                    temp_key = max(
                        matching_temp_keys,
                        key=lambda k: self._program_prompts[k]["timestamp"]
                    )
                    prompt_data = self._program_prompts[temp_key].copy()
                    prompt_data["llm_response"] = response
                    prompt_data["program_id"] = program_id
                    # Real program_id also records timestamp for easy cleanup
                    prompt_data.setdefault("timestamp", time.time())

                    self._program_prompts[program_id] = prompt_data
                    logger.debug(
                        f"Associated prompt data with program {program_id[:8]} from parent {parent_id[:8]}"
                    )
                else:
                    logger.debug(f"No prompt data found for parent {parent_id[:8]}")

                self._cleanup_prompt_cache_locked()

        except Exception as e:
            logger.warning(f"Failed to associate prompt with program {program_id[:8]}: {e}")

    async def _log_prompt_to_database(self, program_id: str, parent_id: str):
        """Persist prompt data to database checkpoint (concurrency-safe)"""
        if not self.log_prompts or self._program_prompts is None:
            return

        try:
            async with self._prompts_lock:
                prompt_data = self._program_prompts.get(program_id, {})
                if prompt_data and 'prompt' in prompt_data:
                    prompt = prompt_data['prompt']
                    response = prompt_data.get('llm_response', '')

                    database_prompt = {
                        'system': prompt.get('system', ''),
                        'user': prompt.get('user', ''),
                        'parent_id': parent_id,
                        'evolution_round': prompt_data.get('evolution_round', 0),
                        'island': prompt_data.get('island', 0),
                        'timestamp': prompt_data.get('timestamp', 0)
                    }

                    self.database.log_prompt(
                        program_id=program_id,
                        template_key='evolution_prompt',
                        prompt=database_prompt,
                        responses=[response] if response else []
                    )

                    logger.debug(f"Logged prompt to database for program {program_id[:8]}")
                else:
                    logger.debug(f"No prompt data to log for program {program_id[:8]}")

        except Exception as e:
            logger.warning(f"Failed to log prompt to database for {program_id[:8]}: {e}")

    def _cleanup_prompt_cache_locked(self):
        """
        Only call when self._prompts_lock is already held.
        - Clean up expired temporary keys
        - Trim by timestamp when total exceeds limit
        """
        if not self.log_prompts or self._program_prompts is None:
            return

        now = time.time()
        # 1) Clean up expired temporary keys
        old_temp_keys = [
            k for k, v in self._program_prompts.items()
            if k.startswith("temp_") and (now - v.get("timestamp", now)) > self._TEMP_PROMPT_TTL_SEC
        ]
        for k in old_temp_keys:
            self._program_prompts.pop(k, None)

        # 2) Total capacity trimming
        if len(self._program_prompts) > self._PROMPT_CACHE_MAX:
            # Sort by timestamp from old to new, delete first N items
            items = [
                (k, v.get("timestamp", 0.0)) for k, v in self._program_prompts.items()
            ]
            items.sort(key=lambda x: x[1])  # Oldest first
            to_delete = len(self._program_prompts) - self._PROMPT_CACHE_MAX
            for i in range(to_delete):
                self._program_prompts.pop(items[i][0], None)

            logger.debug(f"Trimmed prompt cache by {to_delete} items")

    def get_program_prompts(self, program_id: str) -> dict:
        """Get prompt debug data for a program (snapshot)"""
        if not self.log_prompts or self._program_prompts is None:
            return {}
        # No forced locking here, just read snapshot
        return self._program_prompts.get(program_id, {}).copy()

    # ---------------------------
    # Sampling/scoring (logic consistent with iteration.py)
    # ---------------------------


    # @timed_operation("Problem Generation")
    def problem_generator(self) -> Tuple[Dict[str, str], Program]:
        """
        Generate problem - reference iteration.py:52-92

        Returns:
            (prompt_dict, parent_program)
        """
        # Island sampling
        parent, inspirations = self.database.sample(self.config.prompt.num_inspiration_programs)
        parent_artifacts = self.database.get_artifacts(parent.id)
        parent_island = parent.metadata.get("island", self.database.current_island)
        island_top_programs = self.database.get_top_programs(5, island_idx=parent_island)
        island_previous_programs = self.database.get_top_programs(3, island_idx=parent_island)

        def enhance_programs_with_artifacts(programs, include_artifacts):
            # [TODO]: our current code still don't include artifacts in prompt for efficiency (artifacts are not stored). Can try to add it later.
            if not include_artifacts:
                return [p.to_dict() for p in programs]
            enhanced = []
            for p in programs:
                program_dict = p.to_dict()
                artifacts = self.database.get_artifacts(p.id)
                if artifacts:
                    program_dict['artifacts'] = artifacts
                enhanced.append(program_dict)
            return enhanced

        include_top_artifacts = (
            self.config.prompt.include_artifacts_for_all_programs
            or self.config.prompt.include_top_program_artifacts
        )
        include_inspiration_artifacts = (
            self.config.prompt.include_artifacts_for_all_programs
            or self.config.prompt.include_inspiration_artifacts
        )

        prompt = self.prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=enhance_programs_with_artifacts(island_previous_programs, include_top_artifacts),
            top_programs=enhance_programs_with_artifacts(island_top_programs, include_top_artifacts),
            inspirations=enhance_programs_with_artifacts(inspirations, include_inspiration_artifacts),
            language=self.config.language,
            evolution_round=len(self.database.programs),
            diff_based_evolution=self.config.diff_based_evolution,
            program_artifacts=parent_artifacts if parent_artifacts else None,
        )

        # Store debug prompt (temporary key)
        if self.log_prompts and self._program_prompts is not None:
            prompt_data = {
                "parent_id": parent.id,
                "timestamp": time.time(),
                "prompt": prompt,
                "parent_program": parent.code,
                "parent_metrics": parent.metrics,
                "evolution_round": len(self.database.programs),
                "island": parent_island,
            }
            temp_key = f"temp_{parent.id}_{int(time.time() * 1000000)}"
            # Outside async context, avoid await; brief insert here, lock-free write risk is small (exclusive thread)
            self._program_prompts[temp_key] = prompt_data

            # Cleanup (lock-free fast path, real ordered cleanup happens when writing with lock)
            self._cleanup_prompt_cache_locked()

        return prompt, parent

    # Regex that matches EVOLVE-BLOCK markers with Python (#) or C++ (//) comment prefixes
    _EVOLVE_BLOCK_RE = re.compile(
        r"^([ \t]*(?:#|//|/\*)?\s*)EVOLVE-BLOCK-START\s*\n"
        r"(.*?)\n"
        r"([ \t]*(?:#|//|/\*)?\s*)EVOLVE-BLOCK-END\b",
        re.DOTALL | re.MULTILINE,
    )

    @staticmethod
    def _replace_evolve_block(base_program: str, new_block: str) -> str:
        """Replace the EVOLVE-BLOCK in *base_program* with *new_block*."""
        if not base_program or "EVOLVE-BLOCK-START" not in base_program:
            return new_block

        def _repl(m):
            prefix_start = m.group(1)  # e.g. "// " or "# "
            prefix_end = m.group(3)
            return f"{prefix_start}EVOLVE-BLOCK-START\n{new_block}\n{prefix_end}EVOLVE-BLOCK-END"

        return SingleTaskEvolvingGym._EVOLVE_BLOCK_RE.sub(_repl, base_program)

    @staticmethod
    def _extract_evolve_block(program: str) -> str:
        """Extract the content inside the EVOLVE-BLOCK markers."""
        if not program:
            return ""
        match = SingleTaskEvolvingGym._EVOLVE_BLOCK_RE.search(program)
        return match.group(2).strip() if match else program.strip()

    async def _pooled_evaluate(self, child_code: str, program_id: str,
                               timeout_s: float) -> Dict[str, Any]:
        """Run a single evaluation through the shared pool semaphore."""
        async with self._eval_pool:
            return await asyncio.wait_for(
                self.evaluator.evaluate_program(child_code, program_id),
                timeout=timeout_s,
            )

    async def response_scorer(self, response: str, parent_program: Program,
                              n_repeat: int = 1) -> Optional[Result]:
        """
        Score response — concurrency is controlled at the subprocess level
        via _eval_pool (shared semaphore), NOT at the response_scorer level.
        This means n_repeat evaluations compete for the same pool slots as
        initial evaluations from other children.

        Args:
            response: LLM response containing code modifications
            parent_program: Parent program object
            n_repeat: Run evaluation N times and use mean combined_score (mitigates stochasticity)

        Returns:
            Optional[Result] with evaluation results
        """
        result = Result()
        iteration_start = time.time()

        try:
            result.parent = parent_program
            parent_meta = getattr(parent_program, "metadata", {}) or {}
            seen_codes = parent_meta.get("_multistep_seen_codes", [])
            if not isinstance(seen_codes, list):
                seen_codes = []

            def _is_context_lazy(child_code: str) -> bool:
                if check_code_identical(child_code, parent_program.code, self.config.language):
                    logger.info(
                        f"[LazyDetect] identical to PARENT "
                        f"(parent_id={getattr(parent_program, 'id', '?')[:8]}, "
                        f"seen_ctx={len(seen_codes)})"
                    )
                    return True
                for i, prev_code in enumerate(seen_codes):
                    if isinstance(prev_code, str) and prev_code and check_code_identical(
                        child_code, prev_code, self.config.language
                    ):
                        logger.info(
                            f"[LazyDetect] identical to seen_codes[{i}] "
                            f"(seen_ctx={len(seen_codes)}) — trajectory repeat"
                        )
                        return True
                return False

            # Parse response: full-program mode (extract code block, hydrate template)
            logger.debug(f"Response length: {len(response)}")
            logger.debug(f"First 500 chars of response: {response[:500]}")

            code_block = parse_full_rewrite(response, self.config.language)
            if not code_block:
                logger.warning("Full-program mode: no code block found in response")
                result.child_metrics = create_evaluation_metrics(
                    combined_score=ErrorCodes.NO_DIFF_BLOCKS_ERROR,
                    validity=0.0,
                    score_transform_config=self.score_transform_config
                )
                result.artifacts = {
                    "error": "No code block found in LLM response (full-program mode)",
                    "error_type": "no_code_block"
                }
                child_program = Program(
                    id=str(uuid.uuid4()),
                    code="",
                    language=self.config.language,
                    parent_id=parent_program.id,
                    generation=parent_program.generation + 1,
                    metrics=result.child_metrics,
                    iteration_found=len(self.database.programs),
                    metadata={
                        "changes": "No code block found",
                        "parent_metrics": parent_program.metrics,
                    }
                )
                result.child_program = child_program
                return result

            # Hydrate: replace EVOLVE-BLOCK in parent template with new code
            child_code = self._replace_evolve_block(parent_program.code, code_block)
            changes_summary = "Full-program rewrite"

            # NO_VALID_CHANGE if identical to parent or current multistep trajectory context.
            if _is_context_lazy(child_code):
                logger.warning("Full-program mode: code identical to parent")
                result.child_metrics = create_evaluation_metrics(
                    combined_score=ErrorCodes.NO_VALID_CHANGE,
                    validity=0.0,
                    score_transform_config=self.score_transform_config
                )
                result.artifacts = {
                    "error": ErrorMessages.get_message(ErrorCodes.NO_VALID_CHANGE),
                    "error_type": "no_valid_change"
                }
                child_program = Program(
                    id=str(uuid.uuid4()),
                    code=child_code,
                    language=self.config.language,
                    parent_id=parent_program.id,
                    generation=parent_program.generation + 1,
                    metrics=result.child_metrics,
                    iteration_found=len(self.database.programs),
                    metadata={
                        "changes": "Identical to parent",
                        "parent_metrics": parent_program.metrics,
                    }
                )
                result.child_program = child_program
                return result

            # Length limit
            if len(child_code) > self.config.max_code_length:
                logger.warning(
                    f"Generated code exceeds maximum length ({len(child_code)} > {self.config.max_code_length})"
                )
                print(f"[SingleTaskGym] Generated code exceeds maximum length ({len(child_code)} > {self.config.max_code_length})")
                return None

            # Build child program object (before evaluation, for lazy detection)
            child_id = str(uuid.uuid4())
            child_program = Program(
                id=child_id,
                code=child_code,
                language=self.config.language,
                parent_id=parent_program.id,
                generation=parent_program.generation + 1,
                metrics={},  # Empty for now, lazy detection doesn't need evaluation metrics
                iteration_found=len(self.database.programs),
                metadata={
                    "changes": changes_summary,
                    "parent_metrics": parent_program.metrics,
                },
            )

            # Additional lazy output detection (before evaluation to save computation)
            if self.lazy_output_penalty_level >= 2:
                lazy_penalty = self.database.check_lazy_output(
                    child_program, parent_program, self.lazy_output_penalty_level
                )

                if lazy_penalty is not None:
                    # Lazy output, return penalty score directly without evaluation
                    result.child_metrics = {"combined_score": lazy_penalty}
                    result.child_program = child_program
                    result.child_program.metrics = result.child_metrics

                    logger.info(f"Lazy output penalty applied: {lazy_penalty}")
                    return result

            # Not lazy output, continue with normal evaluation
            # All evaluate_program calls go through _pooled_evaluate (shared semaphore).
            timeout_s = getattr(self.config.evaluator, "timeout_s", None) or getattr(self.config.evaluator, "timeout", None) or 1000
            try:
                result.child_metrics = await self._pooled_evaluate(child_code, child_id, timeout_s)
            except asyncio.TimeoutError:
                logger.warning(f"Evaluation timed out for child {child_id[:8]}")
                result.child_metrics = {
                    "combined_score": ErrorCodes.EXECUTION_ERROR,
                    "timeout": True,
                    "error": 0.0
                }
                child_program.metrics = result.child_metrics  # sync metrics or parent lookup crashes
                result.child_program = child_program
                return result

            # Ensure combined_score exists (handle evaluator internal timeout/error cases)
            if "combined_score" not in result.child_metrics:
                logger.warning(f"Missing combined_score in metrics for child {child_id[:8]}, using EXECUTION_ERROR")
                result.child_metrics["combined_score"] = ErrorCodes.EXECUTION_ERROR

            # ── N-repeat mean evaluation (mitigate stochastic evaluator noise) ──
            # Repeats go through the SAME _eval_pool as the initial eval.
            # Total concurrent subprocesses = max_concurrent_evaluations (flat pool).
            first_score = result.child_metrics.get("combined_score", 0.0)
            if n_repeat > 1 and first_score >= 0:
                # Only repeat for valid programs (negative = error codes)
                async def _run_repeat(rep_i: int):
                    try:
                        rep_metrics = await self._pooled_evaluate(
                            child_code, f"{child_id}_r{rep_i+1}", timeout_s
                        )
                        rep_score = rep_metrics.get("combined_score", 0.0)
                        return rep_score if rep_score >= 0 else None
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning(f"Repeat eval {rep_i+1}/{n_repeat-1} failed for {child_id[:8]}: {e}")
                        return None

                rep_results = await asyncio.gather(
                    *[_run_repeat(i) for i in range(n_repeat - 1)]
                )
                repeat_scores = [first_score] + [s for s in rep_results if s is not None]
                mean_score = sum(repeat_scores) / len(repeat_scores)
                result.child_metrics["combined_score_single"] = first_score
                result.child_metrics["combined_score"] = mean_score
                result.child_metrics["eval_n_repeat"] = len(repeat_scores)
                result.child_metrics["eval_n_repeat_scores"] = repeat_scores
                logger.info(
                    f"N-repeat eval for {child_id[:8]}: "
                    f"scores={[round(s,4) for s in repeat_scores]} "
                    f"mean={mean_score:.4f} (single={first_score:.4f})"
                )

            # Artifacts
            artifacts = self.evaluator.get_pending_artifacts(child_id)

            # Update child program object with evaluation results
            child_program.metrics = result.child_metrics
            result.child_program = child_program

            # Associate/persist prompt (concurrency-safe)
            if self.log_prompts:
                await self._associate_prompt_with_program(child_id, parent_program.id, response)
                await self._log_prompt_to_database(child_id, parent_program.id)

            # Result fields
            result.llm_response = response
            result.artifacts = artifacts
            result.iteration_time = time.time() - iteration_start

            pending_envs = getattr(self.evaluator, "_pending_runtime_environments", None)
            if isinstance(pending_envs, dict) and child_id in pending_envs:
                result.runtime_environment_path = pending_envs[child_id]

            logger.info(f"Successfully evaluated child program {child_id}, metrics: {result.child_metrics}")
            return result

        except asyncio.CancelledError:
            logger.info("response_scorer cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error in gym iteration: {e}")
            print(f"[SingleTaskGym] Error in gym iteration: {e}")
            return None

    # === Recording and Visualization (Optional) ===

    def enable_recording(self, output_dir: Optional[str] = "./gym_output") -> None:
        """
        Enable recording and visualization functionality

        Args:
            output_dir: Output directory for recordings (defaults to gym_output)
        """
        from openevolve.evolving_gym.gym_recorder import GymRecorder
        self._recorder = GymRecorder(self, output_dir)
        logger.info("Recording enabled")

    def record_progress(self, training_step: int, **kwargs) -> None:
        """
        Record current gym state (requires initialize() + enable_recording() first)

        Args:
            training_step: Real training step number (required)
            **kwargs: Additional options passed to recorder (save_checkpoint, save_historical_records)
        """
        if self._recorder is None:
            logger.warning("Recording not enabled. Call enable_recording() first.")
            return

        self._recorder.record_step(training_step, **kwargs)

    @property
    def recording_enabled(self) -> bool:
        """Check if recording is enabled"""
        return self._recorder is not None
