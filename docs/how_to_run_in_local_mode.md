
# How to run CoPO in the local mode?

If you want to dive into the code and try to understand what is happening, you can try to set local mode of Ray to True, 
in which case all code will run in a single process so that you can easily set breakpoints to step the code.

However, this will raise problem in native CoPO scripts, since MetaDrive has a strict 
requirement of singleton. Each process can only host one MetaDrive instance, imposed by the simulation engine.

To solve this issue, we need to make several light modifications to the training scripts:
Here is the procedure to setup local mode:

1. Set `config["num_workers] = 1`, indicating that you ask RLLib to only setup one process.
2. Remove `config["evaluation_config]` if any. In CoPO we don't test agents in different environments apart from the training environments, so we don't need to do this step.
3. Remove all `tune.grid_search([...])` code by setting each config with only one value.
4. Set `train(..., local_mode=True)`.


Here is the exemplar code for training IPPO in roundabout environment, provided natively in CoPO codebase:

```python
...
config = dict(
    env=get_rllib_compatible_env(MultiAgentRoundaboutEnv),
    env_config=dict(start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]), ),
    num_gpus=0.25 if args.num_gpus != 0 else 0,
)
train(
    IPPOTrainer,
    exp_name=exp_name,
    keep_checkpoints_num=5,
    stop=stop,
    config=get_ippo_config(config),
    num_gpus=args.num_gpus,
    num_seeds=1,
    test_mode=args.test,
    custom_callback=MultiAgentDrivingCallbacks,
    # local_mode=True
)
```

After the aforementioned modifications, the code becomes:

```python
config = dict(
    env=get_rllib_compatible_env(MultiAgentRoundaboutEnv),
    env_config=dict(start_seed=5000),  # <<<=== Modifified!
    num_gpus=0.25 if args.num_gpus != 0 else 0,
    num_workers=1,  # <<<=== Modifified!
)
train(
    IPPOTrainer,
    exp_name=exp_name,
    keep_checkpoints_num=5,
    stop=stop,
    config=get_ippo_config(config),
    num_gpus=args.num_gpus,
    num_seeds=1,
    test_mode=args.test,
    custom_callback=MultiAgentDrivingCallbacks,
    local_mode=True  # <<<=== Modifified!
)
```

Now you can run the training script with debugger! Please make sure to reset those changes if you want to deploy the script in production. Thanks!