import gymnasium as gym
from gymnasium.core import Env
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env
import eval_highway


# ==================================
#        Main script
# ==================================
model_path = 'highway_agent2.zip'
log_path = 'outputs/highway/runs'
lr = 5e-4
gamma = 0.9
batch_size = 16
n_cpu = 1
num_epochs = 10
timesteps = int(1e5)
ka = 0.001 # 减速惩罚项
kd = 0.01 # 变道惩罚项

class MyWrapper(gym.Wrapper):
    def __init__(self, env: Env, ka, kd):
        super().__init__(env)
        self.ka = ka
        self.kd = kd

    def MyReward(self, action):
        action = torch.tensor(action)
        pa = self.ka * -F.relu(-action[0])
        pd = -abs(self.kd * action[1])
        return pa + pd
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward = reward + self.MyReward(action)
        return obs, reward, done, truncated, info

if __name__ == "__main__":
    train = False
    if train:
        #env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        env = eval_highway.env
        env = MyWrapper(env, ka, kd)
        model = PPO("MlpPolicy",
                    env,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    n_steps=batch_size * 12 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=num_epochs,
                    learning_rate=lr,
                    gamma=0.8,
                    verbose=2,
                    tensorboard_log=log_path)
        # Train the agent
        model.learn(total_timesteps=timesteps)
        # Save the agent
        model.save(model_path)

    model = PPO.load(model_path)
    #env = gym.make("highway-fast-v0")
    env = eval_highway.env
    # obs, info = env.reset()
    # print(obs.shape)
    # print(info)
    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()