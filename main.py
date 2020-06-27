import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import cv2
import time
from dqn import DQNAgent

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel

env_name = "Unity Environment"
train_mode = True

engine_configuration_channel = EngineConfigurationChannel()
env = UnityEnvironment(base_port = 5006, file_name=env_name, side_channels = [engine_configuration_channel])

def preprocess_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    greyscale_array = grayscale_image.reshape(-1, 64, 84, 1)
    return greyscale_array

#Reset the environment
env.reset()

# Set the default brain to work with
group_name = env.get_agent_groups()[0]
group_spec = env.get_agent_group_spec(group_name)

# Set the time scale of the engine
engine_configuration_channel.set_configuration_parameters(time_scale = 3.0)

#Get the state of the agents
step_result = env.get_step_result(group_name)

# Examine the number of observations per Agent
print("Number of observations : ", len(group_spec.observation_shapes))

# Examine the state space for the first observation for all agents
#print("Agent state looks like: \n{}".format(step_result.obs[0]))

# Examine the state space for the first observation for the first agent in rgb
# Shape: 64x84x3 - rgb 
print("Agent state looks like: \n{}".format(step_result.obs[0][0]))

# Agente Shape
print("Agent shape: \n{}".format(np.array(step_result.obs[0][0]).shape))

rgb_array = np.array(step_result.obs[0][0])
greyscale_array = preprocess_image(rgb_array)

action_size = group_spec.discrete_action_branches[0]
image_size = greyscale_array.shape
batch_size = 32
output_dir = 'model_output/gridWorld'
print('Action size: {}'.format(action_size))
number_episodes = 100

agent = DQNAgent(action_size, image_size)

for episode in range(number_episodes):
    env.reset()
    state_image = greyscale_array
    step_result = env.get_step_result(group_name)
    done = False
    episode_rewards = 0
    while not done:
        if group_spec.is_action_discrete():
            action = agent.act(state_image)
            action = np.array([[action]])
        env.set_actions(group_name, action)
        env.step()
        step_result = env.get_step_result(group_name)
        done = step_result.done[0]
        episode_rewards = step_result.reward[0]
        next_state_image = np.array(step_result.obs[0][0])
        next_state_image = preprocess_image(next_state_image)
        
        agent.remember(state_image, action, episode_rewards, next_state_image, done)
        state_image = next_state_image
        
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
        
    if episode % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(episode) + '.hdf5')
    print('Episode', episode)
env.close()
