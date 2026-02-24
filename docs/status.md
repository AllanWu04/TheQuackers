---
layout: default
title: Status
---

# Current Progress

## Project Summary
Our project aimes to develop an autonomous navigation system for a Duckiebot withtin the simulated Duckietown environment.
We strive to create a system with a strong emphasis on safety and adaptability.
With the current environment we are aiming for the agent to reliably preform lane following, staying reasonbly centered throughout the entire route, from start to finish.
A 160x120 RGB visual feed is the primary input for the Duckiebot and provides the agent with critical contect regarding lane positioning.
Based on the camera feed the system produces a continous output in the form of linear and angular velocity commands to control the Duckiebot's movement, allowing it to navigate the field safely and responsivley.
Aiming to transfer the traning agent to the physical Duckiebot we are comparing different RL algorithms to compare and discorve which will give us the best result. 

## Approach
Our Team aims to compare different reinforcement learning approaches for autonomous driving in the Duckiebot simulator.
We decided to implement and compare two distinct policy-gradient methods:
### Proximal Policy Optimization (PPO)
PPO is our primary on-policy method. It uses a clipped objective function to prevent the policy from changing too drastically in a single update, which ensures training stability. The objective function we optimize is:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right]$$

Where $r_t(\theta)$ is the probability ratio, and $\hat{A}_t$ is the estimated advantage at time $t$.

#### 1. Observation & Action Space
* **Observations**: The raw image is resized to $64 \times 64 \times 3$. We utilize `VecTransposeImage` to convert the data to a channel-first format and `VecFrameStack` with `n_stack=4` to allow the agent to perceive temporal information (motion/velocity) from consecutive frames.
* **Actions**: A continuous space representing `[linear_velocity, angular_velocity]`, with both values clipped between $[-1.0, 1.0]$.

#### 2. Reward Function Design
We implemented a multi-faceted reward function to encourage both progress and stability:
* **Progress**: Positive reward based on `velocity_along_path`.
* **Centering**: Penalty based on the `distance_from_path` (using a threshold of $70$ units).
* **Smoothness**: Penalties applied to the magnitude of the yaw velocity to prevent aggressive "wobbling".
* **Termination**: A large penalty ($-100.0$) and episode termination if the `DuckiebotsLineOverlapSensor` detects a collision with lane boundaries.

#### 3. Hyperparameters (Current Configuration)
We selected and tuned the following hyperparameters for PPO to balance training stability with learning speed:

| Hyperparameter | Value |
| :--- | :--- |
| Learning Rate | $2 \times 10^{-4}$ |
| Batch Size | $128$ |
| ent_coef | $0.01$ |
| Total Timesteps | $500,000$ |

##### Tuning Process
Our tuning process focused on the trade-off between exploration and exploitation. We found that a higher entropy coefficient ($ent\_coef$) was necessary because the agent initially struggled to move forward, often getting stuck rotating in circles. By slightly lowering the learning rate, we achieved more consistent convergence across multiple training runs, reducing the likelihood of the "catastrophic forgetting" often seen in RL lane-following tasks.
### Soft Actor-Critic (SAC):
an off-policy maximum-entropy algorithm meant for imprived exploration and sample efficieny in continous control tasks



## Evalution


## Remaining Goals and Challenges
Settle on common parameters for each model, where applicable, to have a better comparison on which preforms the best.
Continously monitor the traning through TensorBoard until we are confident to translate the model onto a physcial Duckiebot.
Run the agent on a physcial DuckieBot and compare the results with the training model.


## Resources Used
- SAC Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
- PPO Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- duckiebotssim : https://gitlab.jblanier.net/sim2real/duckiebotssim/-/tree/master
- AI Tools: We utilized Generative AI tools (Gemini/ChatGPT) to assist in debugging the `duckiebotssim` environment wrappers and to troubleshoot errors within our reinforcement learning training scripts.