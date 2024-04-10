import time
import flappy_bird_gym
import numpy as np

# Initialize Flappy Bird environment
env = flappy_bird_gym.make("FlappyBird-v0")

# Define Q-learning parameters
alpha = 0.2  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial epsilon for exploration
epsilon_min = 0.01  # Minimum epsilon
epsilon_decay = 0.995  # Epsilon decay rate
q_table = np.zeros((2,))  # Q-table for two possible actions: jump or don't jump


# Function to manually track and reward progress
def get_reward(points, done):
    point1 = 0
    point2 = 0
    reward = 0
    # Reward for passing through a pipe
    if points > point1:
        reward = 100
    # Penalize when the bird dies
    #if done:
        #reward = -100
    point1 = point2
    point2 = points
    return reward

# Training loop
for episode in range(1, 10001):  # Train for 1000 episodes
    obs = env.reset()
    total_reward = 0
    points = 0
    should_render = episode % 1000 == 0  # Decide if this episode should be rendered
    while True:
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table)  # Exploit

        # Take action
        next_obs, reward, done, info = env.step(action)
        points += info['score'] 

        # Get reward based on bird's progress
        reward = reward + get_reward(points, done)

        # Update Q-value
        next_state = 0 if next_obs[0] < 0 else 1  # Simple state representation based on bird's y-coordinate
        q_table[action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[action])

        total_reward += reward
        obs = next_obs

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Checking if the player is still alive
        if should_render:
            env.render()  # Render the game
            time.sleep(1/30)
        if done:
            break
        

    # Print total reward per episode
    print(f"Episode {episode}, Total Reward: {total_reward}, Total Points: {points}")

# Evaluation (optional)
# (Code for testing the agent goes here)

env.close()
