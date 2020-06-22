import gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv

class IrrigationEnv(MultiAgentEnv):
    def __init__(self, return_agent_actions = False, part=False):
        self.num_agents = 5
        self.observation_space = gym.spaces.Box(low=0, high=800, shape=(1,))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))

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
        # print(obs)
        # print(self.observation_space)
        return obs, rew, done, info
