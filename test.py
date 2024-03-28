import time
import flappy_bird_gym
env = flappy_bird_gym.make("FlappyBird-v0")

# Define the number of jumps
number_of_jumps = 4
current_jumps = 0

# Define the action for jumping
jump_action = 1

obs = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    # action = env.action_space.sample() # for a random action

    # TODO
    # Build environment:
    # - Vertical and Horizontal distance from pipes
    # - Alive or Dead status
    # Create interface for possible actions:
    # - Jump or don't jump (maybe per time step)
    # Reward:
    # - small positive reward per time step alive
    # - large negative reward for death

    if current_jumps < number_of_jumps:
        action = jump_action  # Make the bird jump
        current_jumps += 1
    else:
        action = 0  # Do nothing (fall)

    # Processing:
    obs, reward, done, info = env.step(action)

    # Rendering the game:
    # (remove this two lines during training)
    env.render()
    time.sleep(1 / 30)  # FPS

    # Checking if the player is still alive
    if done:
        break

env.close()