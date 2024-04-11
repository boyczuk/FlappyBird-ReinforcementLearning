import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import flappy_bird_gym

class DQN(nn.Module): 
    """Deep Q-Network (DQN) class for reinforcement learning.
    
    This class represents a neural network model used in the DQN algorithm for reinforcement learning.
    It takes an input dimension and an output dimension as parameters and constructs a sequential network
    with multiple linear and ReLU layers.
    
    Attributes:
        network (nn.Sequential): The sequential network model.
    
    Methods:
        forward(x): Performs a forward pass through the network.
    """
    def __init__(self, input_dim, output_dim):
        """Initialize the DQN model."""
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        """Perform a forward pass through the network."""
        return self.network(x)

def optimize_model():
    """
    Optimize the model using a batch of transitions from the replay memory.

    This function samples a batch of transitions from the replay memory and uses them to update the Q-values
    of the policy network. The loss is calculated using the mean squared error between the predicted Q-values
    and the target Q-values, which are calculated using the Bellman equation. The optimizer is used to minimize

    the loss and update the weights of the policy network.

    Args:
        None

    Returns:
        None
    """
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    batch = list(zip(*transitions))
    
    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[3])
    next_state_batch = torch.cat(batch[2])
    done_batch = torch.cat(batch[4])
    
    current_q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + (GAMMA * next_q_values * (1 - done_batch))
    
    loss = criterion(current_q_values.squeeze(-1), expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

env = flappy_bird_gym.make("FlappyBird-v0")

state_dim = len(env.observation_space.sample())
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
criterion = nn.MSELoss()
memory = deque(maxlen=10000)

GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
BATCH_SIZE = 128
TARGET_UPDATE = 10

steps_done = 0

def select_action(state):
    """
    Select an action using an epsilon-greedy strategy.
    
    This function selects an action using an epsilon-greedy strategy based on the current state of the environment.
    With probability epsilon, a random action is selected, and with probability 1 - epsilon, the action with the
    highest Q-value is selected.
    
    Args:
        state (torch.Tensor): The current state of the environment.
    
    Returns:
        torch.Tensor: The selected action.    
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_dim)]], dtype=torch.long)


num_episodes = 500 
for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor([state], dtype=torch.float)
    for t in range(1000):
        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], dtype=torch.float)
        
        if not done:
            next_state = torch.tensor([next_state], dtype=torch.float)
        else:
            next_state = None
        
        memory.append((state, action, next_state or state, reward, torch.tensor([done], dtype=torch.uint8)))
        
        state = next_state if next_state is not None else torch.tensor([env.reset()], dtype=torch.float)
        
        optimize_model()
        
        if done:
            break
            
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.close()
