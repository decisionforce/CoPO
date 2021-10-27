# Code for Coordinated Policy Optimization

[**Webpage**](https://decisionforce.github.io/CoPO) | 
[**Code**](https://github.com/decisionforce/CoPO) | 
**Talk** (WIP) | 
[**Paper**](https://arxiv.org/pdf/2110.13827.pdf)

Hi there! This is the source code of the paper “Learning to Simulate Self-driven Particles System with Coordinated Policy Optimization”. 


Please following the tutorial below to kickoff the reproduction of our results.



## Installation

```bash
# Create virtual environment
conda create -n copo python=3.7
conda activate copo

# Install dependency
pip install metadrive-simulator==0.2.3
pip install torch  # Make sure your torch is successfully installed! Especially when using GPU!

# Install environment and algorithm.
cd code
pip install -e .
```



## Training

As a quick start, you can start training CoPO in Intersection environment immediately after installation by running:

```
cd code/copo/
python inter/train_copo_dist.py --exp-name inter_copo_dist 
```

The general way to run training is following:

```
cd code/copo/
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

finally the `EXPNAME` is arbitrary name to denote the experiment (with multiple concurrent trials), such as `roundabout_copo`.


## Visualization

We provide the trained models for all algorithms in all environments. A simple command can bring you the visualization of the behaviors of the populations!

```
cd copo
python vis.py 

# In default, we provide you the CoPO population in Intersection environment. 
# If you want to see others, try:
python vis.py --env round --algo ippo

# Or you can use the native renderer for 3D rendering:
# (Press H to show helper message)
python vis.py --env tollgate --algo cl --use_native_render
```

We hope you enjoy the interesting behaviors learned in this work! 
Please feel free to contact us if you have any questions, thanks! 

## Citation

```latex
@misc{peng2021learning,
      title={Learning to Simulate Self-Driven Particles System with Coordinated Policy Optimization}, 
      author={Zhenghao Peng and Quanyi Li and Ka Ming Hui and Chunxiao Liu and Bolei Zhou},
      year={2021},
      eprint={2110.13827},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```