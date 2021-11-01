from copo.algo_ippo.ippo_cl import IPPOCL, ChangeNCallback
from copo.ccenv import get_change_n_env
from copo.train.train import train
from copo.train.utils import get_train_parser
from copo.utils import get_rllib_compatible_env
from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv
from ray import tune

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"
    stop = int(100_0000)
    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=get_rllib_compatible_env(get_change_n_env(MultiAgentRoundaboutEnv)),
        env_config=dict(
            start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]),
            num_agents=40,
            crash_done=True
        ),

        # ===== Resource =====
        num_gpus=0.25 if args.num_gpus != 0 else 0,
    )

    # Launch training
    train(
        IPPOCL,
        exp_name=exp_name,
        keep_checkpoints_num=3,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        num_seeds=1,
        test_mode=args.test,
        custom_callback=ChangeNCallback,

        # fail_fast='raise',
        # local_mode=True
    )
