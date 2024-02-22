import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np 
import os 

env = gym.make("parking-v0", render_mode="rgb_array")
from stable_baselines3 import PPO
agent = PPO.load('highway_agent.zip')

env.configure({
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
    },

    "duration": 20,  # [s]
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 5,  # [Hz]
})
env.reset()


def load_act_inference():
    #! Load your own act_inference
    #! act_inference= obs: np.ndarray -> action: np.ndarray
    """
    Example:

    act_inference = model.forward 
    return act_inference

    """
    return lambda obs: agent.predict(obs)[0]



act_inference = load_act_inference()

def eval_parking(num_runs,save_path = None):
    eval_results = {}
    list_ep_ret = []
    list_ep_len = []
    for i in range(num_runs):
        ep_ret,ep_len = 0,0
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action = act_inference(obs)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1
        list_ep_ret.append(ep_ret)
        list_ep_len.append(ep_len)
    eval_results['ep_ret'] = np.array(list_ep_ret) 
    eval_results['ep_len'] = np.array(list_ep_len) 

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = save_path + '/eval_results_parking.npy' 
        np.save(file_name,eval_results)
    
    for k,v in eval_results.items():
        print(k,f" Mean: {v.mean().round(4)}, Std: {v.std().round(4)}")



            



if __name__ == '__main__':

    eval_parking(100,save_path = "eval_files/results")
    env.close()