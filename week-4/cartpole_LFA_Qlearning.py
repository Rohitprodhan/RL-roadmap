import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class PolynomialFeatures:
    """
    Transforms 4D state [x, x_dot, theta, theta_dot] into polynomial features.
    With degree=2: [1, x, x_dot, theta, theta_dot, x², x*x_dot, ..., theta_dot²]
    """
    def __init__(self, degree=2):
        self.degree = degree
        self.state_dim = 4
        
        # Normalization bounds for CartPole (map to [-1, 1])
        self.low = np.array([-2.4, -3.0, -0.2095, -3.0])
        self.high = np.array([2.4, 3.0, 0.2095, 3.0])
        self.scale = self.high - self.low
        
        # Calculate total feature dimension
        # For degree 2 with 4 vars: 1 (bias) + 4 (linear) + 10 (quadratic) = 15
        self.n_features = 1 + self.state_dim  # bias + linear
        if degree >= 2:
            # Cross terms and squares: n*(n+1)/2
            self.n_features += self.state_dim * (self.state_dim + 1) // 2
        
    def normalize(self, state):
        """Normalize state to [-1, 1] range"""
        return 2.0 * (state - self.low) / self.scale - 1.0
    
    def transform(self, state):
        """
        Convert state to feature vector phi(s)
        Returns: numpy array of shape (n_features,)
        """
        s = self.normalize(state)
        features = [1.0]  # Bias term
        
        # Linear terms
        features.extend(s)
        
        # Quadratic terms (if degree >= 2)
        if self.degree >= 2:
            for i in range(self.state_dim):
                for j in range(i, self.state_dim):
                    features.append(s[i] * s[j])
        
        return np.array(features, dtype=np.float32)

class ReplayBuffer:
    """Simple experience replay for stabilizing linear FA"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state_feat, action, reward, next_state_feat, done):
        self.buffer.append((state_feat, action, reward, next_state_feat, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)

class LinearQAgent:
    """
    Semi-gradient Q-learning with linear function approximation.
    Q(s, a) = w_a^T * phi(s)
    """
    def __init__(self, n_features, n_actions=2, alpha=0.001, gamma=0.99):
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha          # Learning rate (critical: keep small, e.g., 0.001)
        self.gamma = gamma
        
        # Weight matrix: each action has its own weight vector
        # Initialize small random weights
        self.weights = np.random.randn(n_actions, n_features) * 0.01
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # For tracking
        self.td_errors = []
    
    def q_values(self, state_features):
        """Compute Q(s, a) for all actions"""
        return np.dot(self.weights, state_features)  # Shape: (n_actions,)
    
    def get_action(self, state_features, greedy=False):
        """Epsilon-greedy action selection"""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        q_vals = self.q_values(state_features)
        return np.argmax(q_vals)
    
    def update(self, state_feat, action, reward, next_state_feat, done):
        """
        Semi-gradient Q-learning update:
        w <- w + alpha * [r + gamma*max(Q(s')) - Q(s,a)] * gradient
        where gradient = phi(s) for linear approximation
        """
        # Current Q-value estimate
        current_q = np.dot(self.weights[action], state_feat)
        
        # Compute TD target
        if done:
            td_target = reward
        else:
            next_q_vals = self.q_values(next_state_feat)
            td_target = reward + self.gamma * np.max(next_q_vals)
        
        # TD error
        td_error = td_target - current_q
        self.td_errors.append(abs(td_error))
        
        # Gradient descent on weights
        # Q(s,a) = sum(w_i * phi_i), so dQ/dw_i = phi_i
        self.weights[action] += self.alpha * td_error * state_feat
    
    def update_batch(self, state_feats, actions, rewards, next_state_feats, dones):
        """Batch update for experience replay (more sample efficient)"""
        for i in range(len(actions)):
            self.update(state_feats[i], actions[i], rewards[i], 
                       next_state_feats[i], dones[i])
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_linear_q(n_episodes=2000, use_replay=True):
    """
    Train Linear Q-learning agent on CartPole.
    
    Note: Linear FA with semi-gradient Q-learning can diverge due to the 
    "deadly triad" (off-policy + bootstrapping + function approximation).
    Experience replay and low learning rates help stabilize it.
    """
    env = gym.make("CartPole-v1")
    
    # Feature transformation: degree 2 polynomial
    phi = PolynomialFeatures(degree=2)
    n_features = phi.n_features
    print(f"Using {n_features} features per state (polynomial degree {phi.degree})")
    
    # Agent with conservative learning rate
    agent = LinearQAgent(n_features=n_features, alpha=0.002)
    
    # Replay buffer
    buffer = ReplayBuffer(capacity=5000)
    batch_size = 32
    
    episode_rewards = []
    moving_avg = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        state_feat = phi.transform(state)
        
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state_feat)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_feat = phi.transform(next_state)
            done = terminated or truncated
            
            # Store transition
            if use_replay:
                buffer.push(state_feat, action, reward, next_state_feat, done)
                
                # Update from replay buffer
                if len(buffer) >= batch_size:
                    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                    agent.update_batch(states, actions, rewards, next_states, dones)
            else:
                # Online update (less stable)
                agent.update(state_feat, action, reward, next_state_feat, done)
            
            total_reward += reward
            state_feat = next_state_feat
            
            if done:
                break
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        
        # Calculate moving average
        if episode >= 100:
            avg = np.mean(episode_rewards[-100:])
            moving_avg.append(avg)
        else:
            moving_avg.append(np.mean(episode_rewards))
        
        # Progress reporting
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode+1}: Avg(100)={moving_avg[-1]:.1f}, "
                  f"ε={agent.epsilon:.3f}, TD error(avg)={np.mean(agent.td_errors[-100:]):.4f}")
    
    env.close()
    return agent, phi, episode_rewards, moving_avg

def evaluate(agent, phi, n_episodes=5, render=False):
    """Evaluate the trained linear policy"""
    if render:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1")
    
    rewards = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        total = 0
        states = [state]  # Track trajectory for analysis
        
        while True:
            state_feat = phi.transform(state)
            action = agent.get_action(state_feat, greedy=True)
            state, reward, term, trunc, _ = env.step(action)
            total += reward
            states.append(state)
            
            if term or trunc:
                break
        
        rewards.append(total)
        print(f"Test Episode {ep+1}: Score = {total}")
    
    env.close()
    
    # Show sample trajectory statistics
    print(f"\nAverage test score: {np.mean(rewards):.1f}")
    print(f"Success rate (>400 steps): {np.sum(np.array(rewards) > 400)}/{n_episodes}")
    return rewards

def visualize_weights(agent, phi):
    """Visualize the learned weight vectors for interpretability"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    feature_names = ['bias', 'x', 'x_dot', 'theta', 'theta_dot']
    if phi.degree >= 2:
        # Add quadratic feature names
        vars = ['x', 'x_dot', 'theta', 'theta_dot']
        for i in range(4):
            for j in range(i, 4):
                feature_names.append(f"{vars[i]}*{vars[j]}")
    
    x_pos = np.arange(len(feature_names))
    
    for i, (ax, action_name) in enumerate(zip(axes, ['Left (0)', 'Right (1)'])):
        weights = agent.weights[i]
        colors = ['red' if w < 0 else 'green' for w in weights]
        ax.bar(x_pos, weights, color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'Learned Weights for Action: {action_name}')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Train the agent
    print("Training Linear Q-Learning with Function Approximation...")
    print("Q(s,a) = w_a^T * φ(s) where φ(s) are polynomial features\n")
    
    agent, phi, rewards, moving_avg = train_linear_q(
        n_episodes=2500, 
        use_replay=True  # Set False to see online learning (less stable)
    )
    
    # Plot learning curve
    plt.figure(figsize=(12, 5))
    plt.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    plt.plot(moving_avg, color='red', linewidth=2, label='100-Episode Moving Average')
    plt.axhline(y=475, color='green', linestyle='--', label='Solved (475)')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Linear Function Approximation Q-Learning on CartPole')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Visualize learned weights
    visualize_weights(agent, phi)
    
    # Evaluate
    print("\nEvaluating trained policy...")
    evaluate(agent, phi, n_episodes=10, render=False)