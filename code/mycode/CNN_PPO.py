import numpy as np
import torch
from torch import nn
from torch.nn import functional as F



class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions=2):
        super(PolicyNet, self).__init__()
        self.conv = nn.Sequential( # (batch, 8, 11, 11)
            nn.Conv2d(n_states, 32,3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 128, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3)
        ) # (batch, 128, 1, 1)
        self.fc1 = nn.Linear(128, n_hiddens)
        self.fc_mu = nn.Linear(n_hiddens, n_actions)
        self.fc_std = nn.Linear(n_hiddens, n_actions)

    def forward(self, x): # (batch=1, n_states=8, 11, 11) -> (batch, n_actions), (batch, n_actions)
        x = self.conv(x)
        #print(f'x_shape{x.shape}')
        x = x.squeeze((2,3))
        #x = x.squeeze()
        #print(f'x_shape{x.shape}')
        x = F.relu(self.fc1(x))
        x1 = self.fc_mu(x)
        #print(f'x1_shape{x1.shape}')
        x2 = self.fc_std(x)
        #print(f'x2_shape{x2.shape}')
        mu = 2 * torch.tanh(x1)
        std = F.softplus(x2) + 1e-8
        #print(f'mu_shape{mu.shape}')
        return mu, std



class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.conv = nn.Sequential( # (batch, 8, 11, 11)
            nn.Conv2d(n_states, 32,3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 128, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3)
        ) # (batch, 128, 1, 1)
        self.fc1 = nn.Linear(128, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    def forward(self, x): # (batch=1, 8, 11, 11) -> (batch, 1)
        x = self.conv(x)
        x = x.squeeze((2,3))
        #x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print(f'v_shape{x.shape}')
        return x



class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)

        self.critic = ValueNet(n_states, n_hiddens).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)


        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        #state = torch.tensor(np.array(state)).to(self.device)
        mu, std = self.actor(state)
        action_dict = torch.distributions.normal.Normal(mu, std)
        action = action_dict.sample()
        action = action.squeeze(0)
        #print(action[0])
        #action = torch.tensor([item.cpu().detach().numpy() for item in action])
        return action

    def update(self, transition_dict):
        total_loss = [0,0]

        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).to(self.device)
        # print(f'states{states.shape}')
        # print(f'actions{actions.shape}')
        # print(f'rewards{rewards.shape}')
        # print(f'nextstates{next_states.shape}')
        # print(f'dones{dones.shape}')

        next_states_target = self.critic(next_states).squeeze(1)
        #print(f'r{rewards.shape},n{next_states_target.shape},d{(dones).shape}')
        td_target = rewards + self.gamma * next_states_target * (1 - dones) #(batch,)
        td_target = td_target[:,np.newaxis] #(batch, 1)
        #print(f'tdtarget{td_target.shape}')
        td_value = self.critic(states)
        td_delta = td_value - td_target #(batch, 1)
        #print(f'tddelta{td_delta.shape}')

        td_delta = td_delta.cpu().detach().numpy()
        advantage_list = []
        advantage = 0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device) #(batch, 1)
        #print(advantage)

        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        old_log_prob = action_dists.log_prob(actions) #(batch, 2)
        old_log_prob = old_log_prob.detach()
        #print(f'old_logprob{old_log_prob.shape}')

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_prob = action_dists.log_prob(actions) #(batch, 2)
            #print(f'logprob{log_prob.shape}')
            ratio = torch.exp(log_prob - old_log_prob) #(batch, 2)

            #print(f"ratio{ratio.shape}, ad{advantage.shape}")
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            #print(f'surr2{surr2.shape}')

            actor_loss = torch.mean(-torch.min(surr1, surr2))
            #print(f"critic{self.critic(states).shape}, td{td_target.shape}")
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            total_loss[0] += actor_loss
            total_loss[1] += critic_loss

        total_loss[0]/=self.epochs
        total_loss[1]/=self.epochs
        return total_loss



def MyReward(acc, ka, delta, kd):
    pa = 0
    if acc < 0:
        pa = acc*ka
    pd = -abs(kd*delta)
    return pa + pd



if __name__ == '__main__':
    x = torch.tensor(np.ones([1,8,11,11]) ,dtype=torch.float32)
    print(x.shape)
    policynet = PolicyNet(8,16,2)
    y = policynet(x)
    #print(y)