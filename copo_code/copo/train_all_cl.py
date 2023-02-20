from copo.algo_ippo.ippo_cl import IPPOCL, get_change_n_env, get_change_n_callback
from copo.train.train import train
from copo.train.utils import get_train_parser
from copo.utils import get_rllib_compatible_env
from metadrive.envs.marl_envs import MultiAgentParkingLotEnv, MultiAgentRoundaboutEnv, MultiAgentBottleneckEnv, \
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentIntersectionEnv
from ray import tune

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"
    stop = int(200_0000)
    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=tune.grid_search(
            [
                get_rllib_compatible_env(get_change_n_env(MultiAgentParkingLotEnv)),
                get_rllib_compatible_env(get_change_n_env(MultiAgentIntersectionEnv)),
                get_rllib_compatible_env(get_change_n_env(MultiAgentTollgateEnv)),
                get_rllib_compatible_env(get_change_n_env(MultiAgentBottleneckEnv)),
                get_rllib_compatible_env(get_change_n_env(MultiAgentRoundaboutEnv)),
                get_rllib_compatible_env(get_change_n_env(MultiAgentMetaDrive)),
            ]
        ),
        env_config=dict(start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000])),

        # ===== Resource =====
        # num_workers=1,
        num_gpus=0.5 if args.num_gpus != 0 else 0,
    )

    # Launch training
    train(
        IPPOCL,
        exp_name=exp_name,

        # Note: We should remove this argument for CL! Since the best training performance
        # happens in the early stage of CL training, but the checkpoint at that time has
        # poor test-time performance!
        # keep_checkpoints_num=3,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        num_seeds=1,
        test_mode=args.test,
        custom_callback=get_change_n_callback(stop),

        # fail_fast='raise',
        # local_mode=True
    )
