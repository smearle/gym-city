Copied from [flowersteam's implementation of ALP-GMM](https://github.com/flowersteam/teachDeepRL)




Teacher algorithms for curriculum learning of Deep RL in continuously parameterized environments
==================================
How to teach a Deep RL agent to learn to become good at a skill over a wide range of diverse tasks ? To address this problem
we propose to rely on **teacher algorithms** using **Learning Progress** (LP) as a signal to optimize the sequential selection
of tasks to propose to their DRL Student. To study our proposed teachers, we design two **parameterized BipedalWalker environments**.
Teachers are then tasked to sample a parameter mapped to a *distribution of tasks* on which a task is sampled 
and proposed to the student. The teacher then observes the episodic performance of its student and uses this information to adapt its sampling distribution (see figure below for a simple workflow diagram).

<div style="text-align:center"><img src="teachDRL/graphics/readme_graphics/CTS_framework_pipeline_v2.png"  width="50%" height="50%"/></div>

In this work we present a new algorithm modeling absolute learning progress with Gaussian mixture models (ALP-GMM)
along with existing LP-based algorithms. Using our BipedalWalker environments, we study their efficiency to personalize
a learning curriculum for different learners (embodiments), their robustness to the ratio of learnable/unlearnable
tasks, and their scalability to non-linear and high-dimensional parameter spaces. ALP-GMM, which is conceptually
simple and has very few crucial hyperparameters, opens-up exciting perspectives for various curriculum learning challenges within DRL (domain randomization for Sim2Real transfer, curriculum learning within autonomously discovered task spaces, ...).

Paper: https://arxiv.org/abs/1910.07224

This github repository provides implementations for the following teacher algorithms:
* ALP-GMM, our proposed teacher algorithm
* Robust Intelligent Adaptive Curiosity (RIAC), from [Baranes and Oudeyer, R-IAC: robust intrinsically motivated exploration and active learning.
](https://ieeexplore.ieee.org/document/5342516)
* Covar-GMM, from [Moulin-Frier et al., Self-organization of early vocal development in infants and machines: The role of intrinsic motivation.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3893575/)


## Table of Contents  
**[Installation](#installation)**<br>
**[Testing teachers](#testing-teachers)**<br>
**[Using the Parameterized BW environment](#using-the-parameterized-bw-environment)**<br>
**[Launching experiments on Parameterized BW environment](#launching-experiments-on-parameterized-bw-environment)**<br>
**[Visualizations](#visualizations)**<br>

## Installation

1- Get the repository
```
git clone https://github.com/flowersteam/teachDeepRL
cd teachDeepRL/
```
2- Install it, using Conda for example (use Python >= 3.6)
```
conda create --name teachDRL python=3.6
conda activate teachDRL
pip install -e .
```

## Testing teachers

Test Random, RIAC, ALP-GMM and CovarGMM teachers on a simple toy env
```
cd teachDRL/
python3 toy_env/toy_env.py
```
Gifs of the parameter sampling dynamics of teachers will be created in graphics/toy_env_gifs/

## Using the Parameterized BW environment
In case you want to use our Parameterized BW in your own projects, the following pseudo-code 
shows how to interact with the environment:
```python
import gym
import teachDRL.gym_flowers
import numpy as np
env = gym.make('bipedal-walker-continuous-v0')  # create environment
env.env.my_init({'leg_size': 'default'})  # set walker type within environment, do it once

for nb_episode in range(10):
    # now set the parameters for the procedural generation
    # make sure to do it for each new episode, before reset
    env.set_environment(stump_height=np.random.uniform(0,3),
                        obstacle_spacing=np.random.uniform(0,6)) # Stump Tracks
    #env.set_environment(poly_shape=np.random.uniform(0,4,12))  # Hexagon Tracks
          
    env.reset()
    for i in range(2000):
        obs, rew, done, _ = env.step(env.env.action_space.sample())
        env.render()
```
We implemented two additional walker morphologies, which can be visualized along with the parametric variations
of the environment (Stump Tracks and Hexagon Tracks):
 ```
 python3 test_bipedal_walker_continuous.py
 ```
## Launching experiments on Parameterized BW environment

In the Parameterized BW environment, we use Soft Actor Critic (OpenAI spinningup implementation) as our Deep RL student.


Then you can launch a Teacher-Student run in Stump Tracks:
```
python3 run.py --exp_name test_alpgmm --teacher ALP-GMM --seed 42 --leg_size default --max_stump_h 3.0 --max_obstacle_spacing 6.0
```
Available teachers (-- teacher): ALP-GMM, RIAC, Oracle, Random, Covar-GMM

Available walker morphologies (--leg_size): short, default, quadru

You can also test the quadrupedal walker on Hexagon Tracks:
```
python3 run.py --exp_name test_alpgmm_hexa --teacher ALP-GMM --seed 42 --leg_size quadru -hexa 
```

To run multiple seeds, we recommand to use taskset to bind each process to a single cpu thread, like so:
```
taskset -c 0 python3 run.py --exp_name test_alpgmm_hexa --teacher ALP-GMM --seed 42 --leg_size quadru -hexa &
taskset -c 1 python3 run.py --exp_name test_alpgmm_hexa --teacher ALP-GMM --seed 43 --leg_size quadru -hexa &
```

## Visualizations

### Stump Tracks
We tested the ALP-GMM teacher when paired with 3 different walker morphologies. For each of these walkers we show the
learned walking gates after being trained on a curriculum generated by ALP-GMM. A single run of ALP-GMM allows to train
 Soft Actor-Critic controllers to master a wide range of track distributions.
 
####  ALP-GMM + SAC with short walker (left), default walker (middle) and quadrupedal walker (right)

<p><img src="teachDRL/graphics/readme_graphics/walker_gates/demo_short_stump_gmm_asquad_0.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/demo_default_stump_gmm_asquad_0.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/demo_quadru_stump_gmm_compact_0.gif" width="32%" height="32%"/></p>

<p><img src="teachDRL/graphics/readme_graphics/walker_gates/demo_short_stump_gmm_asquad_3.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/demo_default_stump_gmm_asquad_1.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/demo_quadru_stump_gmm_compact_3.gif" width="32%" height="32%"/></p>

<p><img src="teachDRL/graphics/readme_graphics/walker_gates/demo_short_stump_gmm_asquad_2.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/demo_default_stump_gmm_asquad_2.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/demo_quadru_stump_gmm_compact_2.gif" width="32%" height="32%"/></p>

<p><img src="teachDRL/graphics/readme_graphics/walker_gates/demo_short_stump_gmm_asquad_1.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/demo_default_stump_gmm_asquad_3.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/demo_quadru_stump_gmm_compact_1.gif" width="32%" height="32%"/></p>

The following videos show the evolution of parameter sampling by ALP-GMM for short, default, and quadrupedal walkers.
Learning curricula generated by ALP-GMM are tailored to the capacities of each student it is paired with.

####  ALP-GMM with short walker (left), default walker (middle) and quadrupedal walker (right)
<p><img src="teachDRL/graphics/readme_graphics/GMM_gmmcshortcpu21-0611.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/GMM_gmmcdefaultcpu21-063.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/GMM_gmmclongcpu21-060.gif" width="32%" height="32%"/></p>



### Hexagon Tracks
To assess whether ALP-GMM is able to scale to parameter spaces of higher dimensionality, containing irrelevant
 dimensions, and whose difficulty gradients are non-linear, we performed experiments with quadrupedal walkers on Hexagon Tracks,
  our 12-dimensional parametric Bipedal Walker environment. The following videos shows walking gates learned in a single ALP-GMM run.
<p><img src="teachDRL/graphics/readme_graphics/walker_gates/stump_gmm_demo_26.gif"/></p>

<p><img src="teachDRL/graphics/readme_graphics/walker_gates/stump_gmm_demo_compact_3.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/stump_gmm_demo_compact_8.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/stump_gmm_demo_compact_10.gif" width="32%" height="32%"/></p>

<p><img src="teachDRL/graphics/readme_graphics/walker_gates/stump_gmm_demo_compact_19.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/stump_gmm_demo_compact_48.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/stump_gmm_demo_compact_36.gif" width="32%" height="32%"/></p>

In comparison, when using Random curriculum, most runs end-up with bad performances, like this one:

<p><img src="teachDRL/graphics/readme_graphics/walker_gates/demo_quadru_stump_rand_compact_3.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/demo_quadru_stump_rand_compact_8.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/demo_quadru_stump_rand_compact_10.gif" width="32%" height="32%"/></p>

<p><img src="teachDRL/graphics/readme_graphics/walker_gates/demo_quadru_stump_rand_compact_19.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/demo_quadru_stump_rand_compact_48.gif" width="32%" height="32%"/>
<img src="teachDRL/graphics/readme_graphics/walker_gates/demo_quadru_stump_rand_compact_36.gif" width="32%" height="32%"/></p>
