from tensorforce.agents import Agent
import numpy as np

#Define a environment
class IrrigationEnv():
    def __init__(self, return_agent_actions = False, part=False):
        self.num_agents = 5

    def reset(self):
        obs = {}
        self.dones = set()
        self.water = np.random.uniform(200,800)
        for i in range(self.num_agents):
            obs[i] = np.array([self.water])
        return obs

    def cal_rewards(self, action_dict):
        self.curr_water = self.water
        reward = 0
        for i in range(self.num_agents):
            water_demanded = self.water*action_dict[i][0]

            if self.curr_water == 0:
                reward += 0
                reward -= water_demanded*100
            elif self.curr_water - water_demanded<0:
                water_needed = water_demanded - self.curr_water
                water_withdrawn = self.curr_water
                self.curr_water = 0
                reward += -water_withdrawn**2 + 200*water_withdrawn
                reward -= water_needed*100
            else:
                self.curr_water -= water_demanded
                water_withdrawn = water_demanded
                reward += -water_withdrawn**2 + 200*water_withdrawn

        return reward

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        reward = self.cal_rewards(action_dict)

        for i in range(self.num_agents):

            obs[i], rew[i], done[i], info[i] = np.array([self.curr_water]), reward, True, {}

        done["__all__"] = True
        return obs, rew, done, info

# Get environment obs and act spaces
env = IrrigationEnv()
num_agents = env.num_agents
state_space = dict(type='float', shape=(1,))
action_space = dict(type='float', shape=(1,), min_value=0., max_value=1.)

# Define configuration for agent
config = dict(
            states=state_space,
            actions=action_space,
            network=[
                        dict(type='dense', size=8),
                        dict(type='dense', size=8),
                    ]
)

# Define Agents
agent_list = []
for i in range(num_agents):
    print("Creating Agent {}".format(i))
    agent_list.append(Agent.from_spec(spec='ppo_agent', kwargs=config))

training_iterations = 2000
batch_size = 128

# Create a batch of environments
env_batch = []
for i in range(batch_size):
    env_batch.append(IrrigationEnv())

for i in range(training_iterations):

    # Inititalize some placeholders
    obs_batch = {i:[] for i in range(num_agents)}
    rew_batch = {i:[] for i in range(num_agents)}
    done_batch = {i:[] for i in range(num_agents)}
    action_batch = {b:{} for b in range(batch_size)}

    # Get initial obs for all envs
    for b in range(batch_size):
        obs = env_batch[b].reset()
        for agent_id in range(num_agents):
            obs_batch[agent_id].append(obs[agent_id])

    # Get actions for every agent on a batch of observations
    for agent_id in obs:
        actions = agent_list[agent_id].act(states = obs_batch[agent_id])
        for b in range(batch_size):
            action_batch[b][agent_id] = actions[b]

    # Step all environments based on action batch
    for b in range(batch_size):
        new_obs, rew, dones, info = env_batch[b].step(action_batch[b])
        for agent_id in obs:
            rew_batch[agent_id].append(rew[agent_id])
            done_batch[agent_id].append(dones[agent_id])

    # Call observe to innternalize to experience trajectory
    for agent_id in new_obs:
        agent_list[agent_id].model.observe(reward = rew_batch[agent_id], terminal = done_batch[agent_id])

    # Log rewards
    if i%100==0:
        print("Reward for iteration {} is {}".format(i, np.mean(rew_batch[0])))
