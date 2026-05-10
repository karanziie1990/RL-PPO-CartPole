# PPO from Scratch on CartPole-v1

A from-scratch implementation of Proximal Policy Optimization (PPO) in PyTorch, trained on the CartPole-v1 environment from Gymnasium.

This project was built as part of a reinforcement learning assignment with the goal of understanding how PPO works internally instead of treating RL libraries as black boxes. The implementation covers the full training pipeline including rollout collection, advantage estimation, actor-critic learning, and PPO based policy updates.

---

## Features Implemented

- Actor-Critic architecture
- Rollout collection from environment interaction
- Generalized Advantage Estimation (GAE)
- PPO clipped surrogate objective
- Entropy regularization
- Value function learning
- Minibatch gradient updates
- Training and evaluation on CartPole-v1

---

## Tech Stack

- Python
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

---

## Dependencies

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

Main dependencies used in the project:

```txt
torch
gymnasium
numpy
matplotlib
```

(Optional but recommended)

```txt
jupyter
notebook
```

Because reinforcement learning apparently requires both advanced optimization theory and 17 package installs before a pole learns not to fall over.

---

## Project Structure

```bash
.
├── assignment.ipynb      # Main notebook implementation
├── ppo.py                # PPO training logic
├── models.py             # Actor and Critic networks
├── utils.py              # Helper functions and utilities
├── requirements.txt
└── README.md
```

(Structure may differ slightly depending on how the assignment files are organized.)

---

## PPO Overview

Proximal Policy Optimization (PPO) is a policy-gradient reinforcement learning algorithm introduced by OpenAI. PPO improves training stability by limiting how much the policy is allowed to change during updates.

Instead of making unrestricted policy updates, PPO uses a clipped objective function that prevents overly large updates which can destabilize training.

This project implements PPO manually using PyTorch without relying on prebuilt RL training frameworks such as Stable-Baselines3.

---

## Training Pipeline

The overall training flow is:

1. Collect trajectories from the environment
2. Store states, actions, rewards, and log probabilities
3. Compute discounted returns
4. Estimate advantages using GAE
5. Update actor and critic networks using PPO loss
6. Repeat until convergence

The implementation uses an actor network for policy learning and a critic network for state-value estimation.

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/karanziie1990/RL-PPO-CartPole.git
cd RL-PPO-CartPole
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python ppo.py
```

Or open the notebook:

```bash
jupyter notebook assignment.ipynb
```

---

## Results

The PPO agent successfully learns the CartPole-v1 task over training iterations.

Observed behaviour during training:

- Random and unstable exploration initially
- Gradual improvement in episodic rewards
- Increasingly stable balancing behaviour
- Convergence near the environment reward threshold

Training reward curves and logs can be visualized using Matplotlib.

---

## Key Concepts Explored

This project helped in understanding:

- Policy gradient methods
- PPO clipping mechanism
- Advantage estimation
- Actor-Critic learning
- Reward optimization in RL
- Stability issues during training
- Environment interaction using Gymnasium

More importantly, it clarified how PPO actually behaves during training instead of reducing RL to “import model.train() and hope the graphs go upward.”

---

## Possible Improvements

Some extensions that can be added later:

- Continuous action environments
- TensorBoard logging
- Vectorized environments
- Hyperparameter tuning
- Multi-environment rollout collection
- Comparison against Stable-Baselines3 PPO
- Model checkpoint saving/loading

---

## References

### PPO Paper
https://arxiv.org/abs/1707.06347

### Gymnasium Documentation
https://gymnasium.farama.org/

### PyTorch Documentation
https://pytorch.org/

---

## Author

Karan Ganeshwala  
B.Tech Engineering Science (CS Major)  
IIT Jodhpur
