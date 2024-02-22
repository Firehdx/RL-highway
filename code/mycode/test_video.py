import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
import eval_highway
import eval_intersection
import eval_parking
from stable_baselines3 import PPO

import highway_env

model_path = 'outputs/highway/models.zip'
video_path = 'outputs/highway/videos'

env = eval_highway.env

N_EPISODES = 10  # @param {type: "integer"}

model = PPO.load(model_path, env=env)

env = RecordVideo(env, video_folder=video_path, episode_trigger=lambda e: True)
env.unwrapped.set_record_video_wrapper(env)
for episode in range(N_EPISODES):
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
env.close()