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
- Proximal Policy Optimization (PPO): an on-policy clipped gradient method that emphasizes stable updates
- Soft Actor-Critic (SAC): an off-policy maximum-entropy algorithm meant for imprived exploration and sample efficieny in continous control tasks



## Evalution


## Remaining Goals and Challenges
Settle on common parameters for each model, where applicable, to have a better comparison on which preforms the best.
Continously monitor the traning through TensorBoard until we are confident to translate the model onto a physcial Duckiebot.
Run the agent on a physcial DuckieBot and compare the results with the training model.


## Resources Used
- SAC Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
