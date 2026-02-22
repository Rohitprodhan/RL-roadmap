# The [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) problem.
The original problem have continuous state space, but Tabular Q-learning alorithm is only for discrete state space. Thus we discretize the state space using bins. There are better methods for continuous state space system, but still the implemented Tabular Q Learning agent able to show a decent result (Able to balance the stick for approx 350-450 frames). 


# **Algorithm: Tabular Q-Learning for CartPole**

**1. Initialization**

* Create a **Discretizer**: Divide the continuous world (Position, Velocity, Angle, Angular Velocity) into 14 discrete "bins" per dimension.
* Create a **Q-Table**: A lookup table where every possible state-combination has a score for "Push Left" and "Push Right." Initialize all scores to 0.
* Set **Hyperparameters**: Learning Rate (), Future Discount (), and Exploration Rate ().

**2. Training Loop (Repeat for 8,000 Episodes)**

* **Reset** the environment to the starting upright position.
* **While** the pole has not fallen and time hasn't run out:
1. **Observe** the current state and convert it into discrete "bins."
2. **Select Action**:
* With probability , pick a **random** action (Exploration).
* Otherwise, pick the action with the **highest score** in the Q-Table for this state (Exploitation).


3. **Execute** the action and observe the **New State** and **Reward**.
* *Note: If the pole falls, apply a penalty (Reward = -10).*


4. **Update the Q-Table (The Learning Step)**:


5. **Transition**: Set current state = new state.


* **Decay **: Reduce the exploration rate slightly so the agent relies more on its memory over time.

**3. Evaluation**

* Run the simulation using only the **highest scores** from the Q-Table (no random moves).
* If the average survival time > 475 steps, the system is considered "Solved."

