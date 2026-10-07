import os
import wandb


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def init_wandb_primary(args):
    if not args.use_wandb:
        return None

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            print("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            print("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            print("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    wandb_key = args.wandb_key or os.environ.get("WANDB_API_KEY")  # env keeps the key off the command line
    if (not offline) and wandb_key is not None:
        wandb.login(key=wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    base_group = args.wandb_group
    base_run_name = getattr(args, "wandb_run_name", None) or args.wandb_group
    if args.wandb_random_suffix:
        suffix = wandb.util.generate_id()
        group = f"{base_group}_{suffix}" if base_group else suffix
        run_name = f"{base_run_name}_{suffix}" if base_run_name else group
    else:
        group = base_group
        run_name = base_run_name

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": args.__dict__,
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        print(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    return wandb.run.id


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, wandb_run_id):
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    wandb_key = args.wandb_key or os.environ.get("WANDB_API_KEY")  # env keeps the key off the command line
    if (not offline) and wandb_key is not None:
        wandb.login(key=wandb_key, host=args.wandb_host)

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": args.__dict__,
        "resume": "allow",
        "reinit": True,
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    # Keep generic rollout metrics on rollout/step. Only multistep subtree uses inner-step axis.
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("rollout/multistep/inner_step_cum")
    wandb.define_metric("rollout/multistep/cum_actual_evals")
    # Use cum_actual_evals (honest programs-evaluated count) as the primary x-axis.
    # Falls back gracefully for old runs that don't have chain_actual_k in reward.
    wandb.define_metric("rollout/multistep/*", step_metric="rollout/multistep/cum_actual_evals")
    wandb.define_metric("objective/inner_step")
    wandb.define_metric("objective/cum_actual_evals")
    wandb.define_metric("objective/*", step_metric="objective/cum_actual_evals")
    wandb.define_metric("evolving_gym/cum_actual_evals")
    wandb.define_metric("evolving_gym/*", step_metric="evolving_gym/cum_actual_evals")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")


def finish_wandb(args):
    """Properly finish / flush the wandb run at exit."""
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Failed to finish wandb run")


def get_wandb_offline_dir(args):
    """Get the directory where offline W&B data is stored."""
    if _is_offline_mode(args):
        if args and hasattr(args, "wandb_dir") and args.wandb_dir:
            # Use custom directory if specified
            return args.wandb_dir
        else:
            # Default offline directory is ~/wandb/offline-run-<timestamp>
            # This will be created automatically by wandb
            return os.path.expanduser("~/wandb")
    return None
