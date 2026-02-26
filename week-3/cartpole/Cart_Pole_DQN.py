import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


# --- 1. The Neural Network ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    
def plot_learning_curve(scores):
    plt.figure(figsize=(10,5))
    plt.plot(scores, label='Score per Episode', alpha=0.3, color='blue')
    
    # Calculate moving average (window of 10)
    if len(scores) >= 10:
        moving_avg = np.convolve(scores, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(scores)), moving_avg, label='10-Episode Moving Avg', color='red')
    
    plt.title('DQN Cart-Pole Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()


# --- 2. Hyperparameters ---
LR = 0.001
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
BUFFER_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 10

# --- 3. Setup ---
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict()) # Sync initially
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=BUFFER_SIZE)
epsilon = EPS_START
scores = []

# --- 4. The Training Loop ---
for episode in range(2000):
    state, _ = env.reset()
    total_reward = 0
    
    for t in range(500):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = policy_net(state_t).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Start learning once buffer has enough data
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)

            # Current Q values
            current_q = policy_net(states).gather(1, actions)
            
            # Target Q values (Bellman Equation)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                target_q = rewards + (GAMMA * max_next_q * (1 - dones))

            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break
            
    # Decay epsilon and update target network
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    print(f"Episode {episode}, Score: {total_reward}, Epsilon: {epsilon:.2f}")
    scores.append(total_reward)
env.close()



# Call this after your training loop finishes
plot_learning_curve(scores)