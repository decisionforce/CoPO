from metadrive.envs.marl_envs import MultiAgentParkingLotEnv, MultiAgentRoundaboutEnv, MultiAgentBottleneckEnv, \
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentIntersectionEnv
from ray import tune

from copo.torch_copo.algo_ccppo import CCPPOTrainer, get_ccppo_env
from copo.torch_copo.utils.callbacks import MultiAgentDrivingCallbacks
from copo.torch_copo.utils.train import train
from copo.torch_copo.utils.utils import get_train_parser

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"
    stop = int(100_0000)
    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=tune.grid_search(
            [
                get_ccppo_env(MultiAgentParkingLotEnv),
                # get_ccppo_env(MultiAgentRoundaboutEnv),
                # get_ccppo_env(MultiAgentBottleneckEnv),
                # get_ccppo_env(MultiAgentMetaDrive),
                # get_ccppo_env(MultiAgentTollgateEnv),
                # get_ccppo_env(MultiAgentIntersectionEnv)
            ]
        ),
        # env_config=dict(
        #     start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]),
        # ),

        # ===== Resource =====
        num_gpus=0.25 if args.num_gpus != 0 else 0,

        # ===== MAPPO =====
        counterfactual=True,
        fuse_mode=tune.grid_search(["mf", "concat"]),
        mf_nei_distance=10,
    )

    # Launch training
    train(
        CCPPOTrainer,
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
