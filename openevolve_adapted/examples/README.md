# Tasks

The seven tasks used in the KTPO paper.

| Directory | Task | Domain |
|---|---|---|
| `circle_packing_modular` | Circle packing, n=26 (maximize the sum of radii) | Math |
| `erdos_min_overlap` | Erdős minimum overlap (minimize the overlap bound) | Math |
| `llm_sql` | LLM-SQL column ordering (maximize prefix-cache hit rate) | Systems |
| `cloudcast` | Cloudcast multi-cloud broadcast (minimize transfer cost) | Systems |
| `eplb` | Expert-parallel load balancing for MoE inference | Systems |
| `prism` | PRISM model placement across GPUs | Systems |
| `txn_scheduling` | Transaction scheduling (minimize makespan) | Systems |

Each directory has the same layout:

```
<task>/
├── configs/config_<task>.yaml        # prompt, score transform, pool settings
├── evaluators/evaluator_modular.py   # entry point used during training
└── initial_programs/initial_program.py
```

Task data: `llm_sql/datasets/` and `cloudcast/datasets/` are included. EPLB
needs `eplb/evaluators/expert-load.json` (see the root `README.md`).

See the root `README.md` for how to evaluate a single program.
