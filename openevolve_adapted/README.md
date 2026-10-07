# openevolve (KTPO backend)

Program pool, prompt builder and evaluator harness used by the KTPO training
loop (`slime/`). Adapted from [OpenEvolve](https://github.com/codelion/openevolve)
via [ThetaEvolve](https://github.com/ypwang61/ThetaEvolve). Not a standalone tool.

## Contents

- `openevolve.database.ProgramDatabase`: the program pool, including the pool
  update with improvement-rate eviction.
- `openevolve.evolving_gym.SingleTaskEvolvingGym`: builds prompts from sampled
  parents and evaluates generated programs; driven by `slime`'s rollout loop.
- `openevolve.evaluator`, `openevolve.prompt`, `openevolve.utils`,
  `openevolve.modular_utils`: helpers used by the above.
- `examples/`: the seven paper tasks (initial program, evaluator, config).

The standalone OpenEvolve CLI and LLM client were removed; generation is done
by slime + SGLang.

## Install

```bash
pip install -e .
```

Training instructions are in the repository root `README.md`.
