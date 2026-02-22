import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class StateDiscretizer:
    """
    Converts continuous CartPole state into discrete bins.
    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    """
    def __init__(self, n_bins=12):
        self.n_bins = n_bins
        
        # Define bounds for each dimension (based on CartPole-v1 termination conditions)
        # Position: episode ends at |x| > 2.4, but we give some margin
        self.x_bounds = (-2.4, 2.4)
        
        # Velocity: can be unbounded, but typically stays within [-3, 3]
        self.x_dot_bounds = (-3.0, 3.0)
        
        # Angle: episode ends at |theta| > 0.2095 rad (~12 degrees)
        self.theta_bounds = (-0.2095, 0.2095)
        
        # Angular velocity: typically [-3, 3] rad/s
        self.theta_dot_bounds = (-3.0, 3.0)
        
        self.bounds = [self.x_bounds, self.x_dot_bounds, 
                      self.theta_bounds, self.theta_dot_bounds]
        
        # Create bin edges for each dimension
        self.bin_edges = []
        for low, high in self.bounds:
            edges = np.linspace(low, high, n_bins - 1)
            self.bin_edges.append(edges)
    
    def discretize(self, state):
        """Convert continuous state to discrete tuple (s_0, s_1, s_2, s_3)"""
        discrete_state = []
        for i, (value, edges) in enumerate(zip(state, self.bin_edges)):
            # Clip to bounds to handle out-of-range values
            clipped = np.clip(value, self.bounds[i][0], self.bounds[i][1])
            # Find bin index (0 to n_bins-1)
            bin_idx = np.digitize(clipped, edges)
            discrete_state.append(bin_idx)
        return tuple(discrete_state)

class TabularQLearningAgent:
    def __init__(self, discretizer, alpha=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05):
        self.discretizer = discretizer
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.actions = [0, 1]       # Left, Right
        
        # Q-table: dictionary mapping state_tuple -> array([Q(s,left), Q(s,right)])
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
    
    def get_action(self, state, greedy=False):
        """Epsilon-greedy action selection"""
        disc_state = self.discretizer.discretize(state)
        
        if not greedy and np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.q_table[disc_state])
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update rule: Q(s,a) += alpha * [r + gamma*maxQ(s') - Q(s,a)]"""
        disc_state = self.discretizer.discretize(state)
        disc_next = self.discretizer.discretize(next_state)
        
        current_q = self.q_table[disc_state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[disc_next])
        
        # Update Q-value
        self.q_table[disc_state][action] += self.alpha * (target - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_agent(n_episodes=10000, n_bins=12):
    """
    Train tabular Q-learning agent on CartPole.
    n_bins: granularity of discretization (higher = more precise but slower learning)
    """
    env = gym.make("CartPole-v1")
    discretizer = StateDiscretizer(n_bins=n_bins)
    agent = TabularQLearningAgent(discretizer)
    
    episode_rewards = []
    solved = False
    
    print(f"Training with {n_bins} bins per dimension ({n_bins**4} possible states)...")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Optional: Reward shaping (penalize failure)
            # This helps tabular methods learn faster
            shaped_reward = reward
            if done and steps < 500:
                shaped_reward = -10  # Penalty for dropping the pole
            
            agent.update(state, action, shaped_reward, next_state, done)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
            
            state = next_state
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        
        # Check if solved (CartPole-v1 is considered solved at avg reward > 475 over 100 episodes)
        if episode >= 100 and not solved:
            avg_reward = np.mean(episode_rewards[-100:])
            if avg_reward > 475:
                print(f"\n🎉 Solved at episode {episode}! Average reward: {avg_reward:.1f}")
                solved = True
        
        # Progress reporting
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode+1}, ε={agent.epsilon:.3f}, Last 100 avg: {avg_reward:.1f}")
    
    env.close()
    return agent, episode_rewards

def evaluate_agent(agent, n_episodes=5, render=True):
    """Evaluate the trained agent"""
    if render:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1")
    
    rewards = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.get_action(state, greedy=True)  # No exploration
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
        print(f"Test Episode {ep+1}: Score = {total_reward}")
    
    env.close()
    return rewards

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Train the agent
    # Note: 10-15 bins per dimension usually works best. 
    # Too few (e.g., 5) = too coarse to balance. 
    # Too many (e.g., 50) = takes forever to visit all states.
    agent, rewards = train_agent(n_episodes=8000, n_bins=14)

    #save the Q-table for later reuse
    import pickle
    with open('cartpole_q_table.pkl', 'wb') as f:
        pickle.dump(dict(agent.q_table), f)
    
    # 2. Plot learning curve
    plt.figure(figsize=(12, 5))
    plt.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Moving average
    window = 100
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 
                color='red', label=f'{window}-Episode Moving Avg')
    
    plt.axhline(y=475, color='green', linestyle='--', label='Solved Threshold (475)')
    plt.xlabel('Episode')
    plt.ylabel('Reward (Steps Survived)')
    plt.title('Tabular Q-Learning on CartPole (Discretized State Space)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cartpole_q_learning_training_rewards.png')
    print("\nTraining complete! Reward plot saved as 'cartpole_q_learning_training_rewards.png'.")
    #plt.show()
    
    # 3. Evaluate with visualization
    print("\nEvaluating trained agent...")
    evaluate_agent(agent, n_episodes=5, render=True)