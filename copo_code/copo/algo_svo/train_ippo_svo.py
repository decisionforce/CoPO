from copo.algo_ippo.ippo import IPPOTrainer
from copo.algo_svo.svo_env import get_svo_env
from copo.callbacks import MultiAgentDrivingCallbacks
from copo.train.train import train
from copo.train.utils import get_train_parser
from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv
from ray import tune

SVOPPOTrainer = IPPOTrainer

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"

    stop = int(100_0000)
    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=get_svo_env(MultiAgentRoundaboutEnv),
        env_config=dict(
            start_seed=tune.grid_search([5000, 6000]),
            num_agents=40,
            neighbours_distance=10,
            svo_dist=tune.grid_search(["normal"]),
            svo_normal_std=tune.grid_search([0.3, 0.5, 1.0]),
            crash_done=True,

            # force_svo=tune.grid_search([-1, 0, 0.3, 0.6, 1]),
        ),

        # ===== Resource =====
        # So we need 2 CPUs per trial, 0.25 GPU per trial!
        num_gpus=0.5 if args.num_gpus != 0 else 0,
    )

    # Launch training
    train(
        IPPOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=2,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        num_seeds=1,
        test_mode=args.test,
        custom_callback=MultiAgentDrivingCallbacks,

        # fail_fast='raise',
        # local_mode=True
    )
