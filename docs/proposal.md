---
layout: default
title:  Proposal
---

# DuckyTown Project Proposal

## Summary of the Project
Our project aims to develop an autonomous navigation system for a Duckiebot within the Duckietown simulated environment, with a strong emphasis on safety and adaptability.
At a minimum, the system is designed to reliably perform lane following and remain reasonably centered throughout an entire route from start to finish.
Building on this, a more practical and achievable level of functionality includes recognizing and complying with real-world traffic rules, such as stopping when a stop sign is detected and interpreting road signs and markings to determine the appropriate next action.
Beyond these core capabilities, we also envision extending the system to handle more advanced scenarios involving collision awareness and avoidance, such as reacting to a leading vehicle that suddenly brakes by stopping or swerving when possible, and being aware of pedestrians near or entering the roadway to prevent potential accidents.
The system utilizes a 160x120 RGB visual feed as its primary input, which provides the agent with critical context regarding its lane positioning, detected traffic signage, and potential obstacles. 
Based on this information, the system produces a continuous output in the form of linear and angular velocity commands to control the Duckiebot’s movement, allowing it to navigate safely and responsively.
## Project Goals
### Minimum Goal:
- Perform lane following and stay reasonably centered throughout the whole trip, from start to finish

### Realistic Goal:
- Perform real world traffic laws: stopping when the camera identifies a stop sign, identifying other signs and landmarkings to understand the next action

### Moon Shot Goal:
- Collision detection for obstacles (2 scenarios):
- If following behind another car and they decide to break check, our agent should be able to swerve out of the way or stop before it happens
- Be aware of any pedestrians near the car (sidewalk) so that in an extreme case if a pedestrian happens to get in front of the car the agent should be able to avoid crashing into them
## AI/ML Algorithms
The AI/ML Algorithms that we anticipate on using for DuckyTown will be **Soft Actor-Critic (SAC)** or **Proximal Policy Optimization (PP0)**
reinforcement learning algorithms. These algorithms are both model free, meaning that it does not build an internal model of the environment rules,
but rather reacts to observed rewards and states, and improving its decisions through repeating actions. We believe that when it comes to driving, it would be
better for the AI to learn by experiencing the environment and adapting than internalizing an the environment to predict future states/rewards. Additionally, 
the reason for providing two RL algorithms is due to not fully knowing how each algorithm will work with the Ducky Town Environment. Therefore, potentially experimenting
with both algorithms will help us gain a deeper understanding on why one may be better than another for our specific goals.
## Evaluation Plan
The metric we will use to evaluate the success of our project will rely on a more quantitative evaluation. For example on our minimum goal, once we have completed the necessary training, we will run multiple trials and note how many times and for how long the agent was able to stay centered. Our realistic goal can be measured by determining how many times the agent did a lawful stop or the action taken, based on the sign, was correct. Once these trials are over we can calculate an overall accuracy and determine if further work needs to be done to improve these results. Base line checks would always be if the agent can at least move in each direction and stop after a certain time, if the agent fails to do this then we will reevaluate our implementation. If improvement is required we hope to see at least a 20% increase in accuracy.

To conduct qualitative evaluation we are considering comparing the results of the simulation environment versus real world implementation. After training our agent on a certain map we will also place the agent on an entirely new simulated map. Being faced with a new map simulation will really showcase if our agent was trained will enough to adapt to the new environment and still achieve at least our minimum/realistic goals. If we are given the opportunity we would also like to flash our firmware on the real robot version and see how it interacts with the physical hardware. Simulation can only do so much, faced with a real environment we can be introduced to new "noise" such as the camera being hit by some random lighting change or the motors acting differently. Similarly we know that our agent is successful if our original goals can still be met after adapting to the circumstances.
## AI Tool Usage
Code Debugging & Refinement: AI tools were used to help identify logical errors in our reinforcement learning scripts and to interpret complex error messages from the Duckietown simulator. All suggested fixes were manually reviewed, tested, and integrated by the team.

Technical Documentation & Summarization: AI tools were used to summarize technical papers on algorithms and to assist in structuring Markdown documentation for the project website. All technical decisions and implementations were done by the team. 