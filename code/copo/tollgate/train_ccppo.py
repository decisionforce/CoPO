from copo.algo_ccppo.ccppo import CCTrainerForMAOurEnvironment, get_ccppo_env, register_cc_model
from copo.callbacks import MultiAgentDrivingCallbacks
from copo.train import train, get_train_parser
from metadrive.envs.marl_envs import MultiAgentTollgateEnv
from ray import tune

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"
    register_cc_model()
    stop = int(200_0000)
    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=get_ccppo_env(MultiAgentTollgateEnv),
        env_config=dict(
            start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]),
            num_agents=40,
            crash_done=True,
            neighbours_distance=40,
        ),

        # ===== Resource =====
        num_gpus=0.25 if args.num_gpus != 0 else 0,

        # ===== MAPPO =====
        counterfactual=tune.grid_search([True]),
        fuse_mode=tune.grid_search(["mf"]),
        mf_nei_distance=tune.grid_search([10]),
    )

    # Launch training
    train(
        CCTrainerForMAOurEnvironment,
        exp_name=exp_name,
        keep_checkpoints_num=3,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        num_seeds=1,
        test_mode=args.test,
        custom_callback=MultiAgentDrivingCallbacks,

        # fail_fast='raise',
        # local_mode=True
    )
