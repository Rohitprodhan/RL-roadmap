import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class TileCoder:
    """
    Tile Coding (CMAC) for continuous state spaces.
    Creates multiple overlapping tilings with asymmetric offsets.
    """
    def __init__(self, num_dims, num_tilings=8, num_tiles=8):
        self.num_dims = num_dims
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles  # tiles per dimension per tiling
        
        # Calculate total feature size (no hashing needed for CartPole scale)
        # Each tiling contributes num_tiles^num_dims unique indices
        self.tiling_size = num_tiles ** num_dims
        self.iht_size = num_tilings * self.tiling_size
        
        # Asymmetric offsets: shift each tiling by fraction of tile width
        # This breaks symmetry so different tilings generalize differently
        self.offsets = np.arange(num_tilings) * (1.0 / num_tilings)
        
    def get_features(self, state):
        """
        Convert continuous state (scaled to [0, 1]) to list of active tile indices.
        Returns list of indices (one per tiling).
        """
        # Scale state to [0, num_tiles]
        scaled = state * self.num_tiles
        
        features = []
        for tiling in range(self.num_tilings):
            # Add offset for this tiling and floor to get tile coordinates
            coords = np.floor(scaled + self.offsets[tiling]).astype(int)
            # Clamp to valid range [0, num_tiles-1]
            coords = np.clip(coords, 0, self.num_tiles - 1)
            
            # Convert coords to single index for this tiling (base-num_tiles number)
            index = 0
            for d in range(self.num_dims):
                index += coords[d] * (self.num_tiles ** d)
            
            # Global index = tiling base + local index
            global_idx = tiling * self.tiling_size + index
            features.append(global_idx)
            
        return features  # Length = num_tilings (sparse binary features)

class SARSAAgent:
    """
    Semi-gradient SARSA with linear function approximation.
    Q(s,a) = sum of weights for active tiles (state, action).
    """
    def __init__(self, n_actions, num_dims, num_tilings=8, 
                 alpha=0.1, gamma=1.0, epsilon=1.0):
        self.n_actions = n_actions
        self.alpha = alpha          # Step size (should be divided by num_tilings)
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.num_tilings = num_tilings
        
        # Tile coder for state features
        self.tile_coder = TileCoder(num_dims, num_tilings)
        
        # Weights: one vector per action (shape: [n_actions, iht_size])
        # Initialized to zero (optimistic initialization encourages exploration)
        self.weights = np.zeros((n_actions, self.tile_coder.iht_size))
        
    def q_value(self, state, action):
        """Compute Q(s,a) = w_a^T x(s) by summing active weights."""
        tiles = self.tile_coder.get_features(state)
        return np.sum(self.weights[action, tiles])
    
    def select_action(self, state, greedy=False):
        """Epsilon-greedy action selection."""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        q_values = [self.q_value(state, a) for a in range(self.n_actions)]
        return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, next_action, terminated):
        """
        Semi-gradient SARSA update:
        w <- w + alpha * [r + gamma*Q(s',a') - Q(s,a)] * grad
        where grad = x(s) (1 at active tiles, 0 elsewhere)
        """
        tiles = self.tile_coder.get_features(state)
        current_q = np.sum(self.weights[action, tiles])
        
        # Calculate target (no bootstrap if episode ended)
        if terminated:
            target = reward
        else:
            next_tiles = self.tile_coder.get_features(next_state)
            next_q = np.sum(self.weights[next_action, next_tiles])
            target = reward + self.gamma * next_q
        
        # TD error
        delta = target - current_q
        
        # Update weights: add alpha * delta to each active tile's weight
        # This is equivalent to w += alpha * delta * x(s)
        self.weights[action, tiles] += self.alpha * delta
        
    def decay_epsilon(self, decay=0.995, min_eps=0.01):
        self.epsilon = max(min_eps, self.epsilon * decay)

def train_agent(num_episodes=2000):
    """
    Train SARSA agent on CartPole-v1.
    Returns episode rewards and the trained agent.
    """
    env = gym.make('CartPole-v1')
    
    # CartPole state bounds (for normalization)
    # Position and angle have physical limits; velocities are clipped for stability
    state_low = np.array([-4.8, -4.0, -0.418, -4.0])   # [pos, vel, angle, ang_vel]
    state_high = np.array([4.8, 4.0, 0.418, 4.0])
    
    # Standard practice: alpha divided by number of tilings so that 
    # the effective learning rate across all active features is reasonable
    agent = SARSAAgent(
        n_actions=2,
        num_dims=4,
        num_tilings=8,
        alpha=0.5 / 8,      # 0.5 per update, distributed across 8 tilings
        gamma=1.0,          # Undiscounted episodic task
        epsilon=1.0         # Start fully random
    )
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        # Normalize state to [0, 1] for tile coding
        state = np.clip(state, state_low, state_high)
        state = (state - state_low) / (state_high - state_low)
        
        action = agent.select_action(state)
        total_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Normalize next state
            next_state = np.clip(next_state, state_low, state_high)
            next_state = (next_state - state_low) / (state_high - state_low)
            
            # Select next action (on-policy for SARSA)
            next_action = agent.select_action(next_state)
            
            # Update weights
            agent.update(state, action, reward, next_state, next_action, 
                        terminated or truncated)
            
            total_reward += reward
            state, action = next_state, next_action
        
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        
        # Progress reporting
        if episode % 100 == 0:
            recent_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode:4d} | Avg Reward: {recent_avg:6.2f} | Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    return episode_rewards, agent

def evaluate_agent(agent, num_episodes=5, render=False):
    """Run greedy policy (no exploration) to test performance."""
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    state_low = np.array([-4.8, -4.0, -0.418, -4.0])
    state_high = np.array([4.8, 4.0, 0.418, 4.0])
    
    scores = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = np.clip(state, state_low, state_high)
        state = (state - state_low) / (state_high - state_low)
        
        total = 0
        done = False
        while not done:
            action = agent.select_action(state, greedy=True)
            state, _, term, trunc, _ = env.step(action)
            done = term or trunc
            total += 1
            if not done:
                state = np.clip(state, state_low, state_high)
                state = (state - state_low) / (state_high - state_low)
        scores.append(total)
    
    env.close()
    return scores

if __name__ == "__main__":
    # Train
    print("Training SARSA with Tile Coding on CartPole...")
    rewards, agent = train_agent(num_episodes=1500)
    
    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    
    # Plot moving average
    window = 100
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 
                color='red', label=f'{window}-Episode Moving Average')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('SARSA with Linear Function Approximation (Tile Coding)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sarsa_cartpole_learning_curve.png')  # Save the figure
    plt.show()

    #plot the learned weights
    plt.figure(figsize=(12, 6))
    for action in range(agent.n_actions):
        plt.plot(agent.weights[action], label=f'Action {action}')
    plt.xlabel('Feature')
    plt.ylabel('Weight')
    plt.title('Learned Weights for Each Action')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sarsa_cartpole_weights.png')  # Save the figure
    plt.show()

    # Test final policy
    print("\nTesting greedy policy...")
    test_scores = evaluate_agent(agent, num_episodes=5, render=False)
    print(f"Test episode lengths: {test_scores}")
    print(f"Average test length: {np.mean(test_scores):.1f} (max 500)")