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
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))

        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# Pendulum DDPG
class DDPG():
    def __init__(self, state_dim, action_dim,max_action):
        super(DDPG, self).__init__()
        # Neural Network
        self.actor = Actor(state_dim, action_dim,max_action)
        self.target_actor = Actor(state_dim, action_dim,max_action)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)

        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

        self.memory = ReplayMemory(100000)
        self.replay_memory_size = 100000
        self.discount_factor_g = 0.99          
        self.network_sync_rate = 10          
        self.mini_batch_size = 128           
        self.exploration_noise = 0.1 
        self.tau = 0.01

    # Train the FrozeLake environment
    def train(self, episodes, render=False):
        # Create FrozenLake instance
        env = gym.make('Pendulum-v1', render_mode='human' if render else None)
        # env = env.unwrapped
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]
        min_action = env.action_space.low[0]       

        reward_list = []       
        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    
            episode_reward = 0

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            for j in range(200):
                # # # Select action
                with torch.no_grad():
                    action_tmp = self.select_action(state)
                    # noise = torch.randn_like(action_tmp) * self.exploration_noise
                    noise = np.random.normal(0, self.exploration_noise, size=(action_dim))
                    action = action_tmp + noise
                    # action = action_tmp
                    action = action.clip(min_action,max_action)    
                # epsilon = np.interp(x=i * 200 + j, xp=[0, episodes * 200 / 2],
                #                 fp=[1, 0.02])
                # random_sample = random.random()
                # if random_sample <= epsilon:
                #     action = np.random.uniform(low=-2, high=2, size=action_dim)
                # else:
                #     action = self.select_action(state)

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience into memory
                self.memory.append((state, action, new_state, reward, terminated or truncated))
                if len(self.memory)>self.mini_batch_size:
                    self.optimize()

                # Move to the next state
                state = new_state

                # Increment step counter
                episode_reward = episode_reward + reward

                if terminated or truncated:
                    break

            # print(len(self.memory))
            reward_list.append(episode_reward)
            print(f"Episode: {i+1}, Reward: {round(episode_reward, 3)}")
    

        # Close environment
        env.close()

        # Save policy
        torch.save(self.actor.state_dict(), "test\\pendulum_actor.pt")
        torch.save(self.critic.state_dict(), "test\\pendulum_critic.pt")

        # Create new graph 
        plt.figure(1)
        plt.plot(reward_list)
        
        # Save plots
        plt.savefig("test\\pendulum_ddpg.png")

    # Optimize policy network
    def optimize(self):
        states, actions, next_states, rewards, ends = zip(*self.memory.sample(self.mini_batch_size))
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.vstack(actions))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        ends = torch.FloatTensor(ends).unsqueeze(1)

        # update critic network
        next_action = self.target_actor(next_states)
        target_Q = self.target_critic(next_states, next_action.detach())
        target_Q = rewards + self.discount_factor_g * target_Q * (1 - ends)
        Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor network
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # print(critic_loss,actor_loss)

        # update target critic network
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # update target actor network
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor.forward(state)
        return action.detach().numpy()[0]
    
    def get_action(self,actor_net, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = actor_net.forward(state)
        return action.detach().numpy()[0]

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes,render=True):
        # Create FrozenLake instance
        env = gym.make('Pendulum-v1', render_mode='human' if render else None)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]
        min_action = env.action_space.low[0]

        # Load learned policy
        # Create policy and target policy network
        actor_ddpg = Actor(state_dim,action_dim,max_action)

        # Create critic and target critic network
        critic_ddpg = Critic(state_dim,action_dim)

        actor_ddpg.load_state_dict(torch.load("test\\pendulum_actor.pt"))
        actor_ddpg.eval()    # switch model to evaluation mode

        critic_ddpg.load_state_dict(torch.load("test\\pendulum_critic.pt"))
        critic_ddpg.eval()    # switch model to evaluation mode

        state = env.reset()[0]
        for _ in range(episodes):
            env.render()  
            with torch.no_grad():
                action = self.get_action(actor_ddpg,state)
            state, reward, done, _, _ = env.step(action)
            if done:
            # state, _ = env.reset()
                print("end!")
                break

        env.close()




if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]
    pendulum = DDPG(state_dim,action_dim,max_action)
    # pendulum.train(200)
    pendulum.test(200)


