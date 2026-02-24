---
layout: default
title: Status
---

# Current Progress

## Project Summary
Our project aims to develop an autonomous navigation system for a Duckiebot within the simulated Duckietown environment.
We strive to create a system with a strong emphasis on safety and adaptability.
With the current environment we are aiming for the agent to reliably preform lane following, staying reasonably centered throughout the entire route, from start to finish.
A 160x120 RGB visual feed is the primary input for the Duckiebot and provides the agent with critical context regarding lane positioning.
Based on the camera feed the system produces a continuous output in the form of linear and angular velocity commands to control the Duckiebot's movement, allowing it to navigate the field safely and responsivley.
Aiming to transfer the training agent to the physical Duckiebot we are comparing different RL algorithms to compare and discover which will give us the best result. 

## Approach
Our Team aims to compare different reinforcement learning approaches for autonomous driving in the Duckiebot simulator.
We decided to implement and compare two distinct policy-gradient methods:
### Proximal Policy Optimization (PPO)
PPO is our primary on-policy method. It uses a clipped objective function to prevent the policy from changing too drastically in a single update, which ensures training stability. The objective function we optimize is:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right]$$

Where $r_t(\theta)$ is the probability ratio, and $\hat{A}_t$ is the estimated advantage at time $t$.

### Soft Actor-Critic (SAC):
SAC is our off-policy algorithm chosen for its superior sample efficiency and robustness in continuous action spaces. Unlike PPO, SAC aims to maximize both the expected reward and entropy, which encourages the agent to explore more diverse behaviors and prevents premature convergence to local optima. The objective function for SAC is:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

Where $$\mathcal{H}$$ denotes the entropy and $\alpha$ is the temperature parameter.

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

##### PPO:
We selected and tuned the following hyperparameters for PPO to balance training stability with learning speed:

| Hyperparameter | Value |
| :--- | :--- |
| Learning Rate | $2 \times 10^{-4}$ |
| Batch Size | $128$ |
| ent_coef | $0.01$ |
| Total Timesteps | $500,000$ |

##### Tuning Process
Our tuning process focused on the trade-off between exploration and exploitation. We found that a higher entropy coefficient ($ent\_coef$) was necessary because the agent initially struggled to move forward, often getting stuck rotating in circles. By slightly lowering the learning rate, we achieved more consistent convergence across multiple training runs, reducing the likelihood of the "catastrophic forgetting" often seen in RL lane-following tasks.

##### SAC:
We selected and tuned the following hyperparameters for SAC to balance training stability with learning speed:

| Hyperparameter | Value |
| :--- | :--- |
| Learning Starts | $10,000$ |
| Batch Size | $256$ |
| Train Freq (step)| $1$ |
| Gradient Steps (step) | $1$ |
| Gamma | $0.99$ |
| Tau | $0.005$ |
| Total Timesteps | $500,000$ |

##### Tuning Process
Our SAC tuning focused more on stability and hardware limitations. The default SAC hyperparameters produced the most consistent learning behavior, achieving the highest mean episode rewards among our runs before running out to memory. Yet increasing the timesteps from 100,000 to 500,000 resulted in out-of-memory errors due to the replay buffer storing high-dimensional image observations.

To address this issue we reduced the replay buffer size to 200,000 and a batch size of 256 to balance sample diversity and memory usage. Though this did allow the runs to complete, we observed increase instability in performance over time, some runs even showing rapid degradation.

Overall, SAC demonstrated sensitivity to the replay buffer size and training duration. Our tuning will continue to emphasize computational feasibility while attempting to preserve stable learning dynamics.

## Evaluation
We evaluated our three configurations based on quantitative training logs and qualitative driving performance.

### 1. Quantitative Results
The training progress revealed distinct learning behaviors across the three configurations:

| Metric | Baseline PPO | Tuned PPO | SAC |
| :--- | :--- | :--- | :--- |
| **Final Smoothed Reward** | -434.38 | **-274.10** | -866.27 |
| **Training Steps** | 100,352 | 352,256 | 98,758 |
| **Recovery Status** | Partial / Plateau | **Significant Breakthrough** | Early Stabilization |

#### Baseline PPO Performance:
<img src="imgs/PPO_baseline.png" alt="PPO Rew Curve" width="800" height="500">

The **Baseline PPO** showed an early struggle, with the reward dropping nearly to $-750$ within 20k steps. While it recovered to around $-180$ mid-training, it eventually decayed and plateaued at **-434.38**, indicating it failed to find a long-term stable policy for the environment.

#### Tuned PPO Performance:
<img src="imgs/PPO_tuned.png" alt="PPO Tuned Rew Curve" width="800" height="500">  

Our **Tuned PPO** demonstrated a superior recovery capability. After an extensive exploration phase that dipped to $-700$, the agent achieved a major breakthrough around **260k steps**. It stabilized at a significantly higher smoothed reward of **-274.10**, proving that our hyperparameter adjustments directly improved the agent's ability to learn from road boundary penalties.

#### SAC Performance:
<img src="imgs/SAC_plot.png" alt="SAC Rew Curve" width="800" height="500">

The **SAC** agent initially struggled with higher entropy exploration, with rewards dropping to nearly $-1000$ around 60k steps. However, it showed a clear upward trend in the final 30k steps, reaching **-866.27**. While the reward is lower than PPO at this stage, the trajectory suggests potential for continued improvement with more training steps.

### 2. Qualitative Results
We analyzed the agent's behavior through visual captures to identify failure modes and successes.

![Baseline_PPO_gif](imgs/PPO_baseline_resized.gif) 

**Baseline PPO (Stagnation Mode)**: We observed that the Baseline PPO agent developed a "safe" but useless strategy of **spinning in place**. By doing so, it avoids crossing the lane boundaries and incurring the heavy $-100.0$ penalty, but it fails to make any forward progress.

![Baseline_PPO_gif](imgs/PPO_tuned_resized.gif)

**Tuned PPO (Success Mode)**: 
* **Dynamic Recovery**: Unlike the baseline models, the Tuned PPO agent demonstrates a sophisticated understanding of the environment by making continuous, fine-grained steering adjustments to stay centered.
* **Effective Progress**: After reaching the convergence point, the agent overcame the "stagnation" trap; it actively moves forward with linear velocity while successfully interpreting the $64 \times 64$ RGB input to anticipate upcoming curves.
* **Stability**: The agent maintains a stable trajectory even as training progresses, proving that our tuning effectively balanced exploration with exploitation.

![Baseline_PPO_gif](imgs/SAC_resized.gif)

**SAC (Exploration Mode)**: The SAC agent currently exhibits **wandering behavior**, moving near the starting area but without a clear sense of direction. While it is more active than the Baseline PPO, it has not yet learned to correlate visual inputs with the long-term goal of lane following.



## Remaining Goals and Challenges
Settle on common parameters for each model, where applicable, to have a better comparison on which preforms the best.
Continously monitor the traning through TensorBoard until we are confident to translate the model onto a physical Duckiebot.
Run the agent on a physcial DuckieBot and compare the results with the training model.


## Resources Used
- SAC Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
- PPO Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- duckiebotssim : https://gitlab.jblanier.net/sim2real/duckiebotssim/-/tree/master
- AI Tools: We utilized Generative AI tools (Gemini/ChatGPT) to assist in debugging the `duckiebotssim` environment wrappers and to troubleshoot errors within our reinforcement learning training scripts.


## Progress Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/-cZzBRPxu5M?si=gbzZBSAsnnrsB-rR" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>