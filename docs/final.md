---
layout: default
title: Final Report
---


# Final Report

## Project Summary
Our project aims to develop an autonomous navigation system for a Duckiebot within the simulated Duckietown environment.
We strive to create a system with a strong emphasis on safety and adaptability.
With the current environment we are aiming for the agent to reliably preform lane following, staying reasonably centered throughout the entire route, from start to finish.
A 160x120 RGB visual feed is the primary input for the Duckiebot and provides the agent with critical context regarding lane positioning.
Based on the camera feed the system produces a continuous output in the form of linear and angular velocity commands to control the Duckiebot's movement, allowing it to navigate the field safely and responsivley.
Aiming to transfer the training agent to the physical Duckiebot we are comparing different RL algorithms to compare and discover which will give us the best result. 

---

## Approach

Our team implemented and compared two distinct reinforcement learning approaches for autonomous driving in the Duckiebot simulator to identify the most robust solution for continuous action spaces. We focused on comparing an on-policy method, **Proximal Policy Optimization (PPO)**, with an off-policy method, **Soft Actor-Critic (SAC)**.

### Proximal Policy Optimization (PPO)
PPO was our primary method, chosen for its training stability. It utilizes a clipped objective function to prevent the policy from changing too drastically in a single update:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right]$$

Where $r_t(\theta)$ is the probability ratio, and $\hat{A}_t$ is the estimated advantage at time $t$.

### Soft Actor-Critic (SAC)
SAC was chosen for its superior sample efficiency in continuous action spaces. Unlike PPO, SAC aims to maximize both the expected reward and entropy to encourage exploration and prevent premature convergence:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

Where $\mathcal{H}$ denotes the entropy and $\alpha$ is the temperature parameter.

---

### Implementation Details

#### 1. Observation & Action Space
* **Observations**: Raw images are resized and normalized to $64 \times 64 \times 3$. We utilize `VecTransposeImage` to convert data to a channel-first format and `VecFrameStack` with `n_stack=4` to allow the agent to perceive temporal information (motion and velocity) from consecutive frames.
* **Actions**: A continuous space representing `[linear_velocity, angular_velocity]`, with both values clipped between $[-1.0, 1.0]$.

#### 2. Advanced Reward Shaping
We implemented a custom `ImageWrapper` to refine the reward signal, transitioning from basic penalties to a sophisticated multi-weighted function:
* **Progress (Weight: 3.0)**: Positive reward based on `forward_vel` to encourage forward movement and overcome the "stagnation trap" of spinning in place.
* **Alignment (Weight: 1.5)**: Penalty based on `yaw_vel` to keep the Duckiebot aligned with the lane center.
* **Smoothness (Weight: 0.8)**: Penalty applied to the magnitude of steering changes (`turn_jerk`) to prevent aggressive "wobbling."
* **Spin Penalty (Weight: 0.4)**: Specifically penalizes high angular velocity to discourage the agent from simply rotating to avoid collision rewards.
* **Momentum Bonus**: A $+0.2$ bonus applied when `forward_vel > 0.3` and `abs(yaw_vel) < 0.3` to encourage stable, high-speed lane following.
* **Termination**: A large penalty ($-5.0$ to $-100.0$ depending on the version) and episode termination upon collision with lane boundaries.


### Hyperparameter Configurations

We evolved our hyperparameters across multiple training runs to find the best balance between exploration and stability. Our tuning process was split between optimizing PPO for long-term navigation and managing memory constraints for SAC.

#### 1. Proximal Policy Optimization (PPO)
Our final PPO configuration focused on increasing the batch size and learning rate to handle the high-dimensional input from the $64 \times 64 \times 3$ image wrapper.

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Learning Rate** | $3 \times 10^{-4}$ | Increased from $1 \times 10^{-4}$ to accelerate convergence with custom rewards. |
| **n_steps** | $2048$ | Increased from $1024$ to provide more stable gradient estimates per update. |
| **Batch Size** | $128$ | Increased from $64$ to improve update stability in continuous action spaces. |
| **ent_coef** | $0.05$ | Set high to ensure the agent explored forward movement instead of spinning. |
| **Total Timesteps** | $2,000,000$ | Extended training duration to ensure behavior stabilization. |

**PPO Tuning Process:**
The primary challenge with PPO was the "stagnation trap," where the agent would spin in place to avoid collision penalties. By increasing the entropy coefficient ($ent\_coef$) to $0.05$ and adjusting the reward weights for forward progress, we forced the agent to explore the lane boundaries more effectively.

#### 2. Soft Actor-Critic (SAC)
Our final PPO configuration focused on increasing the batch size and learning rate to handle the high-dimensional input from the $64 \times 64 \times 3$ image wrapper.

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Learning Rate** | $3 \times 10^{-4}$ | Increased from $1 \times 10^{-4}$ to accelerate convergence with custom rewards. |
| **n_steps** | $2048$ | Increased from $1024$ to provide more stable gradient estimates per update. |
| **Batch Size** | $128$ | Increased from $64$ to improve update stability in continuous action spaces. |
| **ent_coef** | $0.05$ | Set high to ensure the agent explored forward movement instead of spinning. |
| **Total Timesteps** | $2,000,000$ | Extended training duration to ensure behavior stabilization. |

**PPO Tuning Process:**
The primary challenge with PPO was the "stagnation trap," where the agent would spin in place to avoid collision penalties. By increasing the entropy coefficient ($ent\_coef$) to $0.05$ and adjusting the reward weights for forward progress, we forced the agent to explore the lane boundaries more effectively.

---

## Evaluation

We evaluated our PPO and SAC configurations across four distinct training attempts for each algorithm, measuring performance through quantitative reward metrics and qualitative behavioral analysis in the DuckieTown simulator.

### 1. Proximal Policy Optimization (PPO) Evaluation

The PPO training evolved from baseline establishment to late-stage refinement using custom reward functions.

#### Early/Mid-Stage Training (Baseline and Initial Tuning)
* **Model 1 (Baseline)**: Utilizing 100,000 training steps and default hyperparameters, this model achieved a reward value of **-468.3351**.
* **Model 2 (Initial Tuning)**: Training was extended to 500,000 steps with a learning rate of $1 \times 10^{-4}$ and an entropy coefficient of $0.05$. This configuration resulted in a reward value of **-456.0184**.

<img src="imgs/PPO_1&2.png" alt="PPO Rew Curve" width="800" height="500">

|  ![PPO1_gif](imgs/PPO1.gif)  | ![PPO2_gif](imgs/PPO2.gif) |
|:----------------------------:|:--------------------------:|
|      Model 1 (Baseline)      |          Model 2 (Initial Tuning)          |


#### Late-Stage Training (Reward Refinement)
* **Model 3 (Extended Training with Simple Reward)**: With 1,000,000 training steps and the "simple" reward function, performance improved to a reward of **-141.3634**.
* **Model 4 (Optimized Custom Reward)**: Maintaining the same hyperparameters as Model 3 but implementing a "custom" reward function, the agent achieved a breakthrough positive reward of **83.9519**. This version demonstrated the most stable and centered lane-following behavior.
<img src="imgs/PPO_3&4.png" alt="PPO Rew Curve" width="800" height="500">

|            ![PPO3_gif](imgs/PPO3.gif)            | ![PPO4_gif](imgs/PPO4.gif) |
|:------------------------------------------------:|:--------------------------:|
|  Model 3 (Extended Training with Simple Reward)  | Model 4 (Optimized Custom Reward) |

### 2. Soft Actor-Critic (SAC) Evaluation

SAC evaluation focused on overcoming hardware constraints and exploration issues through seeding and hyperparameter modification.

#### Early Stage (Baseline and Modified Parameters)
* **Model 1 (Default SAC)**: Using default parameters over 100,000 timesteps, the agent achieved an initial reward of **-418.5**.
* **Model 2 (Modified Hyperparameters)**: Training was extended to 500,000 timesteps with an increased learning rate of $3 \times 10^{-4}$ and a buffer size of 200,000. This configuration initially struggled, resulting in a reward of **-2551**.

<img src="imgs/SAC_1&2.png" alt="PPO Rew Curve" width="800" height="500">

|  ![SAC1_gif](imgs/SAC1.gif)  | ![SAC2_gif](imgs/SAC2.gif) |
|:----------------------------:|:--------------------------:|
|    Model 1 (Default SAC)     |  Model 2 (Modified Hyperparameters)  |
#### Mid and Final Stages (Seeding and Custom Rewards)
* **Model 3 (Mid-Stage SEEDING)**: To stabilize training, domain randomization and camera location randomization were disabled. With a smaller buffer size (50,000) and lower learning rate ($1 \times 10^{-4}$), the model reached a reward of **1514**.
* **Model 4 (Final Stage Custom Reward)**: Building on the seeded environment, a custom reward function was applied. There are two runs. The "Gray" run achieved the highest SAC reward of **2493**, while the "Orange" run (with modified forward weighting) achieved **1403**.

<img src="imgs/SAC_3.png" alt="PPO Rew Curve" width="800" height="500">
<img src="imgs/SAC_4&5.png" alt="PPO Rew Curve" width="800" height="500">

| ![SAC3_gif](imgs/SAC3.gif)  |       ![SAC4_gif](imgs/SAC4.gif)        |          ![SAC5_gif](imgs/SAC5.gif)          |
|:---------------------------:|:---------------------------------------:|:--------------------------------------------:|
| Model 3 (Mid-Stage SEEDING) | Model 4 (Final Stage Custom Reward / Gray ) | Model 4 (Final Stage Custom Reward / Orange) |

### Final Comparison and Insights
* **Behavioral Progression**: Baseline models often fell into the "stagnation trap," spinning in place to avoid penalties. Policy refinement and custom reward shaping were essential to achieving consistent forward progress and trajectory smoothness.
* **Training Stability**: PPO demonstrated more consistent convergence during late-stage refinement compared to SAC, which showed higher sensitivity to buffer sizes and learning starts.
* **Performance Peak**: Model 4 for PPO (using 1,000,000 steps and custom rewards) and Model 3 for SAC (Mid-Stage SEEDING) represented the most successful configurations for autonomous navigation.

---

## Resources Used
- SAC Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
- PPO Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- duckiebotssim : https://gitlab.jblanier.net/sim2real/duckiebotssim/-/tree/master
- AI Tools: We utilized Generative AI tools (Gemini/ChatGPT) to assist in debugging the `duckiebotssim` environment wrappers and to troubleshoot errors within our reinforcement learning training scripts.


## Progress Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/-cZzBRPxu5M?si=gbzZBSAsnnrsB-rR" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>