import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.tensorboard_utils import _TensorboardAdapter
from slime.utils.wandb_utils import finish_wandb, init_wandb_primary


class _NoOpActor:
    """Stub for --openai-api-base mode: all training calls become no-ops."""
    def set_rollout_manager(self, rm): pass
    def update_weights(self): pass
    def async_train(self, *a, **kw): return ray.put(0)
    def save_model(self, *a): pass
    def offload(self): pass


def train(args):
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    if args.use_tensorboard:
        _TensorboardAdapter(args)

    # OpenAI inference mode: skip training model creation entirely.
    # _NoOpActor + args.offload_train/offload_rollout=False +
    # args.use_critic=False makes the
    # rest of the loop work unchanged — no extra guards needed.
    if getattr(args, "openai_api_base", None):
        actor_model, critic_model = _NoOpActor(), None
        if args.start_rollout_id is None:
            args.start_rollout_id = 0
    else:
        actor_model, critic_model = create_training_models(args, pgs, wandb_run_id=wandb_run_id)

    # create the rollout manager, with sglang engines inside.
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)

    actor_model.set_rollout_manager(rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())
        ray.get(rollout_manager.onload_weights.remote())

    # Always do initial update_weights to establish engine connections.
    # This also syncs weights from training actor → sglang inference engine.
    actor_model.update_weights()

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    _sample_budget = getattr(args, "sample_budget", None)
    if _sample_budget:
        print(f"[SampleBudget] Run will stop when cum_actual_evals >= {_sample_budget} "
              f"(num_rollout={args.num_rollout} is an upper bound)")
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # TODO extract the duplicated eval logic
        if args.eval_interval is not None and rollout_id == 0:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        # Train step: internally handles debug_rollout_only mode
        # Multi-epoch (n_train_epochs_per_rollout) is handled INSIDE train_actor()
        # to avoid recomputing log probs — only the gradient step is repeated.
        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))


        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            actor_model.save_model(rollout_id)
            if args.use_critic:
                critic_model.save_model(rollout_id)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))
            elif args.evolving_gym:
                ray.get(rollout_manager.save.remote(rollout_id))
            else :
                assert False, "None of args.rollout_global_dataset, args.evolving_gym is set."

        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())

        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

        # ── Sample-budget early exit ──────────────────────────────────────────
        # Check AFTER eval so the final eval still runs at budget boundary.
        if _sample_budget:
            _cum_evals = ray.get(rollout_manager.get_cum_actual_evals.remote())
            if _cum_evals >= _sample_budget:
                print(
                    f"[SampleBudget] ✅ Budget exhausted: cum_actual_evals={_cum_evals} "
                    f">= sample_budget={_sample_budget}  (rollout_id={rollout_id}). Stopping.",
                    flush=True,
                )
                break
            else:
                print(
                    f"[SampleBudget] {_cum_evals}/{_sample_budget} evals used "
                    f"({100*_cum_evals/_sample_budget:.1f}%)  rollout={rollout_id}",
                    flush=True,
                )

    finish_wandb(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
