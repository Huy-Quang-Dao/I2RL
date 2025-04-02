import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.autograd.set_detect_anomaly(True)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.action_probs = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.action_probs(x),dim=-1)

        return x


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Memory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self):
        return self.memory

    def __len__(self):
        return len(self.memory)
    
    def clear(self):
        self.memory.clear()


# Cartpole DDPG
class PPO():
    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        # Neural Network
        self.actor = Actor(state_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic = Critic(state_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

        self.memory = Memory(200)
        self.memory_size = 200
        self.discount_factor = 0.98                         
        self.epoch = 25
        self.lamda = 0.95
        self.eps = 0.2
        self.max_grad_norm = 0.5
        self.batch_size = 32

    # Train the FrozeLake environment
    def train(self, episodes, render=False):
        # Create FrozenLake instance
        env = gym.make('CartPole-v1', render_mode='human' if render else None)
        # env = env.unwrapped
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        reward_list = []       
        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    
            episode_reward = 0

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):
                action = self.select_action(state)

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience into memory
                self.memory.append((state, action, new_state, reward, terminated or truncated))

                # Move to the next state
                state = new_state

                # Increment step counter
                episode_reward = episode_reward + reward

            if len(self.memory)>self.batch_size:
                self.optimize()        

        # Close environment
        env.close()

        # Save policy
        torch.save(self.actor.state_dict(), "PPO\\cartpole_actor2.pt")
        torch.save(self.critic.state_dict(), "PPO\\cartpole_critic2.pt")

        # Create new graph 
        plt.figure(1)
        plt.plot(reward_list)
        
        # Save plots
        plt.savefig("PPO\\cartpole_ppo2.png")

    # Optimize policy network
    def optimize(self):
   
        for _ in range(self.epoch):
            mini_batches = random.sample(self.memory.sample(), self.batch_size)

            states, actions, next_states, rewards, ends = zip(*mini_batches)
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions))
            next_states = torch.FloatTensor(np.array(next_states))
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            ends = torch.FloatTensor(ends).unsqueeze(1)

            # computing TD-error
            target_val = self.critic(next_states)
            v_target_val = rewards + self.discount_factor * target_val * (1 - ends)
            v_val = self.critic(states)
            td_error = v_target_val - v_val

            # advantages = torch.zeros_like(td_error)
            # advantage = 0
            # for i in reversed(range(len(td_error))):
            #     advantage = td_error[i] + self.discount_factor * self.lamda * advantage * (1 - ends[i])
            #     advantages[i] = advantage

            advantages = torch.zeros_like(td_error)
            advantages[-1] = td_error[-1]
            for i in reversed(range(len(td_error)-1)):
                advantages[i] = td_error[i] + self.discount_factor * self.lamda * advantages[i+1] * (1 - ends[i])

            action_probs = self.actor(states).detach()
            old_distribution = torch.distributions.Categorical(probs=action_probs)
            old_log_distribution = old_distribution.log_prob(actions)

            # computing new distribution
            action_probs = self.actor(states)
            new_distribution = torch.distributions.Categorical(probs=action_probs)
            log_distribution = new_distribution.log_prob(actions)
            ratio_distribution = torch.exp(log_distribution - old_log_distribution)

            # computing loss
            term1 = ratio_distribution * advantages.detach()
            term2 = torch.clamp(ratio_distribution, 1 - self.eps, 1 + self.eps) * advantages.detach()
            new_v_val = self.critic(states)
            actor_loss = torch.mean(-torch.min(term1, term2))
            critic_loss = torch.mean(F.mse_loss(new_v_val, v_target_val.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.memory.clear()


    def select_action(self, state):
        with torch.no_grad():
            action_probs = self.actor(torch.tensor(state).unsqueeze(0))
            distribution = torch.distributions.Categorical(probs=action_probs)
            action = distribution.sample()
        return action.item() 
    
    # def set_action(self, actor,state):
    #     with torch.no_grad():
    #         action_probs = actor(torch.tensor(state).unsqueeze(0))
    #         distribution = torch.distributions.Categorical(probs=action_probs)
    #         action = distribution.sample()
    #     return action.item() 
    

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes,render=True):
        # Create FrozenLake instance
        env = gym.make('CartPole-v1', render_mode='human' if render else None)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Load learned policy
        # Create policy and target policy network
        actor_ppo = Actor(state_dim,action_dim)

        # Create critic and target critic network
        critic_ppo = Critic(state_dim)

        actor_ppo.load_state_dict(torch.load("PPO\\cartpole_actor2.pt"))
        actor_ppo.eval()    # switch model to evaluation mode

        critic_ppo.load_state_dict(torch.load("PPO\\cartpole_critic2.pt"))
        critic_ppo.eval()    # switch model to evaluation mode

        state = env.reset()[0]
        for _ in range(episodes):
            env.render()  
            with torch.no_grad():
                action_probs = actor_ppo(torch.tensor(state).unsqueeze(0))
                distribution = torch.distributions.Categorical(probs=action_probs)
                action = distribution.sample().item()
            state, reward, done, _, _ = env.step(action)
            if done:
            # state, _ = env.reset()
                print("end!")
                break

        env.close()



if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    cartpole = PPO(state_dim,action_dim)
    cartpole.train(500)
    # cartpole.test(200)












