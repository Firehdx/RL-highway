import numpy as np
import matplotlib.pyplot as plt
#import gym
from gymnasium.wrappers import RecordVideo
import torch
from CNN_PPO import PPO, MyReward
import eval_intersection
import eval_highway



model_path = ''
output_path = ''
video_path = ''
task = 'highway'
train = True

num_episodes = 2e5
epochs = 10  # 一个episode训练的轮次
disp_freq = 10 # 显示reward频率
gamma = 0.9
actor_lr = 1e-3
critic_lr = 5e-4
lmbda = 0.95  # GAE的缩放因子
clp = 0.2  # CLIP range
n_hiddens = 16

ka = 0.001 # 减速惩罚项
kd = 0.01 # 变道惩罚项


if __name__ == '__main__':
    env = eval_highway.env

    if train:
        device = torch.device('cuda') if torch.cuda.is_available() \
                                    else torch.device('cpu')
        num_episodes = int(num_episodes)
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]
        return_list = []

        agent = PPO(n_states=n_states,
                    n_hiddens=n_hiddens,
                    n_actions=n_actions,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    lmbda = lmbda,
                    epochs = epochs,
                    eps = clp,
                    gamma=gamma,
                    device = device
                    )

        f = open(output_path+'log.txt', 'w')

        for i in range(num_episodes):
            
            state = env.reset()[0]
            done = False
            truncated = False
            episode_return = 0

            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
            }

            while not (done or truncated):
                #go straight for information
                action = agent.take_action(state).cpu().numpy()
                #print(action)
                next_state, reward, done, truncated, _  = env.step(action)
                reward = reward + MyReward(action[0], ka, action[1], kd)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                #print(state == next_state)
                state = next_state
                episode_return += reward

                # if i%10==0:
                #     env.render()
            return_list.append(episode_return)

            actor_loss, critic_loss = agent.update(transition_dict)

            if i%disp_freq == 0:
                message = f'episode:{i}, reward:{episode_return}, actor loss:{actor_loss}, critic loss:{critic_loss}'
                f.write(message+'\n')
                print(message)

        f.close()
        plt.plot(return_list)
        plt.title('return')
        plt.show()
        torch.save(agent, model_path+'PPO_'+task+f'_lr={actor_lr}-{critic_lr}_lambda={lmbda}_gamma={gamma}_clp={clp}.pth')

    else:
        agent = torch.load('PPO_intersection_lr=0.001-0.001_lambda=0.95_gamma=0.95_clp=0.2.pth')
        env = RecordVideo(env, video_folder=video_path, episode_trigger=lambda e: True)
        env.unwrapped.set_record_video_wrapper(env)
        for episode in range(3):
            state, info = env.reset()
            done = truncated = False
            while not (done or truncated):
                action = agent.take_action(state).cpu().numpy()
                state, reward, done, truncated, info = env.step(action)
                env.render()
        env.close()