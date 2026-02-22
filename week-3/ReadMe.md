# Week 3: Model-Free Prediction and Control

## Overview
This week covers learning from experience without knowing the environment dynamics (the transition probability).

And implementation of [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) problem using Q learning algorithm. (in the '[cartpole](\cartpole)' forlder)

## Core Algorithms

### Monte Carlo (MC)
- Learn from **complete episodes** using average returns
- High variance, zero bias
- Update: $V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$

### Temporal Difference (TD)
- Learn from **incomplete episodes** using bootstrapping
- Lower variance than MC, biased
- Update: $V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$

### SARSA (On-Policy)
- Updates using the **actual next action** taken
- Conservative: avoids risky paths
- Update: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$

### Q-Learning (Off-Policy)
- Updates using the **best possible next action**
- Optimal: learns the best policy despite exploration
- Update: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

## Key Difference: SARSA vs Q-Learning

**Cliff Walking Example** (4×12 grid, goal at end, cliff in middle):

| Algorithm | Path | Avg Reward |
|-----------|------|-----------|
| **SARSA** | Around cliff (safe) | ~-27 |
| **Q-Learning** | Along cliff edge (optimal) | ~-17 |

## When to Use

- **SARSA**: Safety-critical systems (autonomous vehicles, robots)
- **Q-Learning**: Optimal policies needed (game AI, planning)

## Implementation
See `week3.ipynb` for:
- CliffWalkingEnv class
- SARSAAgent and QLearningAgent
- 500-episode training comparison
- Performance plots and policy visualization

## Hyperparameters
- **α** (learning rate): 0.1
- **γ** (discount): 0.99
- **ε** (exploration): 0.1 (ε-greedy)

## Quick Start
1. Run `week3.ipynb` cells sequentially
2. Observe learning curves and compare SARSA vs Q-Learning
3. Visualize learned policies (how they navigate the cliff)
4. Move to more complex [cart pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) problem in  the '[cartpole](\cartpole) folder.
