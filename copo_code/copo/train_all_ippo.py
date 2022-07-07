from copo.callbacks import MultiAgentDrivingCallbacks
from copo.algo_ippo.ippo import IPPOTrainer
from copo.train.train import train
from copo.train.utils import get_train_parser
from copo.utils import get_rllib_compatible_env
from metadrive.envs.marl_envs import MultiAgentParkingLotEnv, MultiAgentRoundaboutEnv, MultiAgentBottleneckEnv, \
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentIntersectionEnv
from ray import tune

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"

    # Setup config
    stop = int(100_0000)

    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=tune.grid_search(
            [
                get_rllib_compatible_env(MultiAgentParkingLotEnv),
                get_rllib_compatible_env(MultiAgentIntersectionEnv),
                get_rllib_compatible_env(MultiAgentTollgateEnv),
                get_rllib_compatible_env(MultiAgentBottleneckEnv),
                get_rllib_compatible_env(MultiAgentRoundaboutEnv),
                get_rllib_compatible_env(MultiAgentMetaDrive),
            ]
        ),
        env_config=dict(start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]), ),

        # ===== Resource =====
        # So we need 2 CPUs per trial, 0.25 GPU per trial!
        num_gpus=0.5 if args.num_gpus != 0 else 0,
    )

    # Launch training
    train(
        IPPOTrainer,
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
