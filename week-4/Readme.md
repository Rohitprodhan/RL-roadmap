# Week 4: Linear Function Approximation

## Overview

So far whatever algorithms we have learned are only for discrete state space. In this week's material we'll learn algorithms for continuous state space. Thus, we'll move from tabular methods to handling continuous and large state spaces using **linear function approximation** (We can also approximate non linear functions, but linear function approximation is easiest to start with and also very important for conceptual understanding). We move from lookup tables to parameterized weight vectors, enabling generalization across similar states.

**Prerequisite**: Week 1 (CartPole with Tabular Q-Learning)  
**Resource**: Sutton & Barto, Ch. 9 (9.1–9.4), Ch. 10 (10.1–10.2)

---

## Core Concepts

### 1. What is Function Approximation?

Function approximation is the use of parameterized functional forms to represent value functions or policies, rather than explicit tables (lookup arrays). Instead of storing values for every individual state $Q(s,a)$, we learn a parameterized function $\hat{q}(s, a, \mathbf{w})$ where $\mathbf{w}$ is a weight vector that generalizes across similar states.

There are two main approaches to function approximation in RL:

1. **Value Function Approximation** (This week): We approximate the value function $\hat{v}(s, \mathbf{w})$ or $\hat{q}(s, a, \mathbf{w})$ using parameters $\mathbf{w}$, then derive the policy greedily from these values (e.g., $\epsilon$-greedy on $\hat{q}$). This extends the methods we've learned (TD, SARSA, Q-learning) to continuous spaces.

2. **Policy Approximation** (Next week): We approximate the policy function $\pi(a|s, \mathbf{\theta})$ directly using parameters $\mathbf{\theta}$, without explicitly representing value functions. This is known as **Policy Gradient** methods, where we optimize the policy parameters to maximize expected return.

This week focuses on **Value Function Approximation** with linear function approximation.

### 2. Objective Function: Mean Squared Value Error

Unlike tabular methods that update individual entries, function approximation updates weights by following the gradient of an **objective function**. We minimize the **Mean Squared Value Error (VE)** between our approximate values and the true values:

For state-values:

$$
\overline{VE}(\mathbf{w}) = \sum_{s} \mu(s) \left[v_\pi(s) - \hat{v}(s, \mathbf{w})\right]^2
$$


For action-values (used in control):

$$
\overline{VE}(\mathbf{w}) = \sum_{s} \mu(s) \sum_{a} \pi(a|s) \left[q_\pi(s,a) - \hat{q}(s,a, \mathbf{w})\right]^2
$$


Where $\mu(s)$ is the state distribution under policy $\pi$. We perform stochastic gradient descent on this objective, which yields the weight update rules used in our algorithms.

### 3. Linear Approximation Form
Approximate action-values as a linear function of features:

$$
\hat{q}(s, a, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s, a)
$$


Where:
- $\mathbf{w} \in \mathbb{R}^d$: weight vector (learned parameters)
- $\mathbf{x}(s, a) \in \mathbb{R}^d$: feature vector (hand-engineered)

### 4. The Semi-Gradient Update
Using stochastic gradient descent on the VE objective:


$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{1}{2}\alpha \nabla \overline{VE}(\mathbf{w}_t) = \mathbf{w}_t + \alpha \delta_t \nabla \hat{q}(s_t, a_t, \mathbf{w}_t)
$$


For linear FA, $\nabla \hat{q}(s, a, \mathbf{w}) = \mathbf{x}(s, a)$, giving us:

$$
\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \mathbf{x}(s, a)
$$


**TD Error ($\delta$) varies by algorithm:**
- **TD(0)**: $\delta = r + \gamma \hat{q}(s', \mathbf{w}) - \hat{q}(s, \mathbf{w})$
- **SARSA**: $\delta = r + \gamma \hat{q}(s', a', \mathbf{w}) - \hat{q}(s, a, \mathbf{w})$
- **Q-learning**: $\delta = r + \gamma \max_{a'} \hat{q}(s', a', \mathbf{w}) - \hat{q}(s, a, \mathbf{w})$$

*Called "semi-gradient" because we ignore the gradient of the bootstrapped target (which itself depends on $\mathbf{w}$).*

### 5. Feature Construction
Since the approximator is linear, feature engineering is critical:
- **Polynomials**: $\phi(s) = [1, s_1, s_2, s_1^2, s_1s_2, ...]$
- **Tile Coding**: Overlapping grid tiles (binary features)
- **Radial Basis Functions**: Gaussian kernels centered at prototype points

### 6. Convergence Reality Check
| Algorithm | Linear FA Stability |
|-----------|-------------------|
| TD(0) | ✅ Converges (to near-optimal) |
| SARSA | ⚠️ No guarantees (oscillations possible) |
| Q-learning | ❌ Can diverge (unstable) |

This introduces the **Deadly Triad**: combining [Function Approximation] + [Bootstrapping] + [Off-policy Learning] can cause divergence. Q-learning with linear FA hits all three.

---

## Implementation Roadmap

### CartPole Implementation Details

We implement the CartPole problem using **Linear Function Approximation** with **SARSA** ('cartpole_LFA_SARSA.py')and **Q-learning** ('cartpole_LFA_Qlearning.py'), where features are constructed as **second-order combinations of the four original states**.

**Original State** (4D): $s = [x, \dot{x}, \theta, \dot{\theta}]$  
(Cart position, cart velocity, pole angle, pole angular velocity)

**Feature Engineering**: Second-order polynomial features including:
- **Bias term**: $1$ (constant)
- **Linear terms**: $x, \dot{x}, \theta, \dot{\theta}$ (4 features)
- **Quadratic terms**: $x^2, \dot{x}^2, \theta^2, \dot{\theta}^2$ (4 features)  
- **Cross terms**: $x\dot{x}, x\theta, x\dot{\theta}, \dot{x}\theta, \dot{x}\dot{\theta}, \theta\dot{\theta}$ (6 features)

**Total**: **15 features**

**Algorithms implemented:**
- **Semi-gradient SARSA with LFA**: On-policy control
- **Semi-gradient Q-learning with LFA**: Off-policy control (watch for instability)

### Minimum Viable Architecture
```python
class LinearFA:
    def __init__(self, n_actions, n_features):
        self.w = np.zeros((n_actions, n_features))
    
    def q_value(self, features, action):
        return self.w[action] @ features
    
    def update(self, features, action, target, alpha):
        # Semi-gradient update
        prediction = self.q_value(features, action)
        self.w[action] += alpha * (target - prediction) * features