# Code for Coordinated Policy Optimization

[**Webpage**](https://decisionforce.github.io/CoPO) | [**Code**](https://github.com/decisionforce/CoPO) |  [**Paper**](https://arxiv.org/pdf/2110.13827.pdf) | [**Talk (English)**](https://youtu.be/sOw43l8lwxE) | [**Talk (Chinese)**](https://www.bilibili.com/video/BV1gr4y1C7Ab)



```diff
Changelog:
+ June 22, 2022: Update README to include FAQ, update evaluate population script
```



Hi there! This is the source code of the paper “Learning to Simulate Self-driven Particles System with Coordinated Policy Optimization”. 


Please following the tutorial below to kickoff the reproduction of our results.



## Installation

```bash
# Create virtual environment
conda create -n copo python=3.7
conda activate copo

# Install latest MetaDrive
git clone https://github.com/decisionforce/metadrive
cd metadrive
pip install -e .

# Install dependency
pip install torch  # Make sure your torch is successfully installed! Especially when using GPU!
cd copo_code
pip install -e .
```

Please install latest [MetaDrive](https://github.com/decisionforce/metadrive).
Note that we do not support `gym>=0.20.0`. In `setup.py` we specify we are using `gym==0.19.0`.


## Training

As a quick start, you can start training CoPO in Intersection environment immediately after installation by running:

```
cd copo_code/copo/
python inter/train_copo_dist.py --exp-name inter_copo_dist 
```

Please visit each training script to adjust the hyper-parameters. 
The general way to run training is following:

```
cd copo_code/copo/
python ENV/train_ALGO.py --exp-name EXPNAME 
```

Here `ENV` refers to the shorthand of environments:

```
round  # Roundabout
inter  # Intersection
bottle  # Bottleneck
parking  # Parking Lot
tollgate  # Tollgate
```

and `ALGO` is the shorthand for algorithms:

```
ippo  # Individual Policy Optimization
ccppo  # Mean Field Policy Optimization
cl  # Curriculum Learning
copo_dist  # Coordinated Policy Optimiztion (Ours)
copo_dist_cc  # Coordinated Policy Optimiztion with Centralized Critics
```

finally the `EXPNAME` is arbitrary name to denote the experiment. One experiment contains multiple concurrent trials with different random seeds or sweeps of hyper-parameter. By default the exp name is `TEST`.

If you want to verify the code in details, e.g. through setting breakpoints, you can to learn how to run CoPO in the local mode in [FAQ section](#faq). 

## Visualization

We provide the trained models for all algorithms in all environments. A simple command can bring you the visualization of the behaviors of the populations!

```
cd copo_code/copo
python vis.py 

# In default, we provide you the CoPO population in Intersection environment. 
# If you want to see others, try:
python vis.py --env round --algo ippo

# Or you can use the native renderer for 3D rendering:
#  Press H to show helper message
#  Press Q to switch to third-person view
python vis.py --env tollgate --algo cl --use_native_render
```

We hope you enjoy the interesting behaviors learned in this work! 
Please feel free to contact us if you have any questions, thanks! 

You can also try [vis_from_checkpoint.py](copo_code/copo/vis_from_checkpoint.py) that
can directly load model from RLLib checkpoint.


## FAQ


### How to run CoPO in the local mode?

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


## Citation

```latex
@article{peng2021learning,
  title={Learning to Simulate Self-Driven Particles System with Coordinated Policy Optimization},
  author={Peng, Zhenghao and Hui, Ka Ming and Liu, Chunxiao and Zhou, Bolei},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

