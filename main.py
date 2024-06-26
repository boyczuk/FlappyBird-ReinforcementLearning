'''
This project implements Q-Learning on Flappy Bird using flappy_bird_gym. 

Authors: Adlai Bridson-Boyczuk, Alice Slabosz, and Muhammad Ibrahim
Date: 04/12/2024
'''
# Import libraries
import numpy as np
import flappy_bird_gym
import matplotlib.pyplot as plt
import time

# Initialize Flappy Bird instance
env = flappy_bird_gym.make("FlappyBird-v0")

# Initialize Q-table
num_bins = 15
bin_edges = np.linspace(-0.5, 0.5, num_bins + 1)[1:-1]
q_table = np.zeros((num_bins, num_bins, 2))

# Learning parameters
learning_rate = 0.01
discount_factor = 0.99
global epsilon
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01

# Initialize lists
episode_rewards = []
episode_lengths = []
episode_avg = []
episode_scores = []

def discretize_state(obs):
    """
    Convert continuous state into discrete bins.

    This function discretizes the continuous state space into discrete bins.

    Args:
        obs (tuple): The current state of the environment.

    Returns:
        tuple: The discretized state.
    """
    vertical_dist, horizontal_dist = obs[0], obs[1]
    vertical_idx = np.digitize(vertical_dist, bin_edges)
    horizontal_idx = np.digitize(horizontal_dist, bin_edges)
    return vertical_idx, horizontal_idx

def choose_action(state):
    """
    Epsilon-greedy action selection.

    This function selects an action using an epsilon-greedy strategy based on the current state of the environment.
    With probability epsilon, a random action is selected, and with probability 1 - epsilon, the action with the
    highest Q-value is selected.

    Args:
        state (tuple): The current state of the environment.
    
    Returns:
        int: The selected action.
    """
    global epsilon
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])
    
def render_episode(env):
    """
    Render an episode for visualization.
    
    This function renders an episode using the environment's render function.
    
    Args:
        env (gym.Env): The environment to render.
    
    Returns:
        None
    """
    obs = env.reset()
    done = False
    while not done:
        env.render()
        state = discretize_state(obs)
        action = choose_action(state)
        obs, reward, done, _ = env.step(action)

def update_q_table(state, action, reward, next_state):
    """
    Update Q-value using the Q-learning formula.

    This function updates the Q-value of the state-action pair using the Q-learning formula.

    Args:
        state (tuple): The current state of the environment.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (tuple): The next state of the environment.

    Returns:
        None
    """
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + discount_factor * q_table[next_state + (best_next_action,)]
    td_error = td_target - q_table[state + (action,)]
    q_table[state + (action,)] += learning_rate * td_error

# Initialize loop parameters to get results
num_episodes = 20000
all_scores = []
all_average_scores = []
rendered = False

# Loop games of flappy bird with Q-learning and print results for analysis
for episode in range(num_episodes): # Loop over the episodes
    obs = env.reset() # Reset the environment
    state = discretize_state(obs) # Discretize the state
    
    # Initialize variables
    total_reward = 0 
    steps = 0
    score = 0
    previous_score = 0
    done = False 

    while not done: # Loop over the steps
        action = choose_action(state) # Choose an action
        next_obs, reward, done, info = env.step(action) # Take a step
        next_state = discretize_state(next_obs) # Discretize the next state

        # Update the score and reward
        score = info['score']
        if score > previous_score:
            reward = 100  # Reward for passing a pipe
            previous_score = score
        elif done:
            reward = -200  # Adjusted penalty for dying to be less severe
        else:
            reward = -1 # encourage exploration
        
        update_q_table(state, action, reward, next_state) 
        
        # Update the state
        state = next_state
        total_reward += reward
        steps += 1
        episode_rewards.append(total_reward)

    # Append the total reward and episode length
    episode_scores.append(score)
    all_scores.append(score)
    average_score = np.mean(episode_scores)
    all_average_scores.append(average_score)

    # Decrease epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Every 1000 episodes, reduce learning rate
    if episode % 1000 == 0 and episode != 0:
        learning_rate *= 0.9 
        print(f"Reduced learning rate to: {learning_rate}")

    # Every 100 episodes, print results and append the average rewards then clear
    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Average Score: {average_score:.2f}, Score: {score}, Total Reward: {total_reward}, Episode Length: {steps}")
        temp = np.mean(episode_rewards)
        episode_avg.append(temp)
        episode_rewards.clear

# Plotting score per episode and average score per episode
plt.figure(figsize=(10, 5))
plt.plot(all_scores, label='Score per Episode')
plt.plot(all_average_scores, label='Average Score')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Score and Average Score per Episode')
plt.legend()
plt.show()

# Initialize parameters for scatterplot
step = 100  # Number of episodes to average over
eps = []
for i in range(num_episodes):
    eps.append(i)
episodes = eps[0::step]

# Implement scatterplot for rewards per episode by averaging each 100 episodes for clearer graphing
plt.figure(figsize=(12, 6))
plt.scatter(episodes, episode_avg, s=10, alpha=0.6)  # Small markers, slightly transparent
plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.title('Average Total Rewards Per Every {} Episodes'.format(step))
plt.grid(True)
plt.show()

# Close Flappy Bird instance
env.close()
