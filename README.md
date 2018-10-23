# Project 2: Continuous Control
## Saminda Abeyruwan

### Introduction

In this project, we have successfully trained and evaluated an agent to solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

[![Alt text](https://img.youtube.com/vi/_X8TD39DvNA/0.jpg)](https://www.youtube.com/watch?v=_X8TD39DvNA)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of our agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, angular velocities of the arm, and so on. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

We have trained and evaluated our agent the second version contains 20 identical agents, each with its own copy of the environment. We have used second environment to use the distributed algorithm, Continuous Synchronous Advantage Actor Critic ([A2C](https://arxiv.org/pdf/1602.01783v1.pdf)). To solve the problem, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Therefore,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
* This yields an **average score** for each episode (where the average is over all 20 agents).

The agent has solved the problem within __134 episodes__, and the weights are saved in [model-A2CAgent.bin](model-A2CAgent.bin).


### Report

The [Report.md](Report.md) contains the detail description of the methodology used in the development of the agent.  

### Training

	python3 ac2_agent.py --file_name=path/to/Reacher.app --train_agent=1 --train_mode=1

The agent was trained on a Mac. 	

### Testing

	python3 ac2_agent.py --file_name=path/to/Reacher.app --train_mode=0
	
Please provide the path of the 20 agent simulator binary or app in the __file\_name__	argument. __train\_mode__=0 runs the simulator in real_time, while, 1, switch to training mode, which runs above real-time. 
