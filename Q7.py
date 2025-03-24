import numpy as np
import gym
import random
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output


from gym import Env
from gym.spaces import Discrete, Box

class MazeEnv(Env):
    def __init__(self):
        super().__init__()
        
        # Define the grid size (5x5 maze)
        self.grid_size = 5
        
        # Define the action space (Up, Down, Left, Right)
        self.action_space = Discrete(4)
        
        # Observation space: Agent's position in the grid
        self.observation_space = Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        
        # Define walls in the maze
        self.walls = [(1, 1), (1, 3), (3, 1), (3, 3)]  # List of wall positions
        
        # Define start and goal positions
        self.start_pos = np.array([0, 0])
        self.goal_pos = np.array([4, 4])
        
        # Reset the environment
        self.state = self.start_pos.copy()

    def step(self, action):
        """Defines how the agent moves in the maze based on actions."""
        x, y = self.state

        if action == 0:  # Move Up
            x -= 1
        elif action == 1:  # Move Down
            x += 1
        elif action == 2:  # Move Left
            y -= 1
        elif action == 3:  # Move Right
            y += 1

        # Ensure agent stays within bounds
        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)

        # Prevent moving into walls
        if (x, y) not in self.walls:
            self.state = np.array([x, y])

        # Check if goal is reached
        if np.array_equal(self.state, self.goal_pos):
            reward = 10  # Large reward for reaching goal
            done = True
        else:
            reward = -1  # Small penalty for each step
            done = False

        return self.state, reward, done, {}

    def reset(self):
        """Resets the environment to the starting position."""
        self.state = self.start_pos.copy()
        return self.state

    def render(self):
        """Prints the current state of the maze."""
        maze = np.full((self.grid_size, self.grid_size), ' ')
        maze[self.goal_pos[0], self.goal_pos[1]] = 'G'
        
        for wall in self.walls:
            maze[wall[0], wall[1]] = 'X'
        
        x, y = self.state
        maze[x, y] = 'A'  # Agent's position
        
        clear_output(wait=True)
        for row in maze:
            print(" ".join(row))
        time.sleep(0.3)


# Initialize Q-table
q_table = np.zeros((5, 5, 4))  # (state_x, state_y, actions)

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration-exploitation trade-off
epsilon_decay = 0.995
min_epsilon = 0.01

# Training parameters
num_episodes = 1000
max_steps = 50

env = MazeEnv()
rewards_per_episode = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state[0], state[1], :])  # Exploit

        next_state, reward, done, _ = env.step(action)

        # Update Q-value using the Q-learning formula
        q_table[state[0], state[1], action] = (1 - alpha) * q_table[state[0], state[1], action] + \
            alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1], :]))

        total_reward += reward
        state = next_state

        if done:
            break

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Store rewards for visualization
    rewards_per_episode.append(total_reward)

    if episode % 100 == 0:
        print(f"Episode {episode}: Reward = {total_reward}")

print("Training complete!")


plt.plot(rewards_per_episode)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()


def test_agent(env, q_table, num_trials=5):
    for trial in range(num_trials):
        state = env.reset()
        done = False
        print(f"Trial {trial+1}:")

        while not done:
            env.render()
            action = np.argmax(q_table[state[0], state[1], :])  # Best action
            state, _, done, _ = env.step(action)

        print("\nGoal Reached!\n")

test_agent(env, q_table)


alpha = 0.2  # Increase learning rate
gamma = 0.95  # Increase future rewards consideration
epsilon_decay = 0.99  # Slower exploration decay
