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
        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))

        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, s, a):
        x = torch.cat((s, a))
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
    def __init__(self):
        super(DDPG, self).__init__()
        # Hyperparameters (adjustable)
        self.learning_rate_a = 0.01         # learning rate actor
        self.learning_rate_c = 0.01         # learning rate critic
        self.discount_factor_g = 0.9         # discount rate (gamma)    
        self.network_sync_rate = 100          # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = 2000       # size of replay memory
        self.mini_batch_size = 32           # size of the training data set sampled from the replay memory
        self.exploration_noise = 0.1

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer_actor = None                # NN Optimizer. Initialize later.
        self.optimizer_critic = None                # NN Optimizer. Initialize later.

    # Train the FrozeLake environment
    def train(self, episodes, render=False):
        # Create FrozenLake instance
        env = gym.make('Pendulum-v1', render_mode='human' if render else None)
        # env = env.unwrapped
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]
        min_action = env.action_space.low[0]
        
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target policy network
        actor_ddpg = Actor(state_dim,action_dim,max_action)
        actor_target = Actor(state_dim,action_dim,max_action)

        # Create critic and target critic network
        critic_ddpg = Critic(state_dim,action_dim)
        critic_target = Critic(state_dim,action_dim)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        actor_target.load_state_dict(actor_ddpg.state_dict())

        # Make the target and critic networks the same (copy weights/biases from one network to the other)
        critic_target.load_state_dict(critic_ddpg.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer_actor = torch.optim.Adam(actor_ddpg.parameters(), lr=self.learning_rate_a)

        # Critic network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer_critic = torch.optim.Adam(critic_ddpg.parameters(), lr=self.learning_rate_c)

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
            
        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):

                # Select action
                with torch.no_grad():
                    action_tmp = self.select_action(actor_ddpg,state)
                    # noise = torch.randn_like(action_tmp) * self.exploration_noise
                    noise = np.random.normal(0, self.exploration_noise, size=(action_dim))
                    action = action_tmp + noise
                    action = action.clip(min_action,max_action)    


                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, actor_ddpg,actor_target, critic_ddpg,critic_target)
                # return print(self.optimize(mini_batch, actor_ddpg,actor_target, critic_ddpg,critic_target))

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    actor_target.load_state_dict(actor_ddpg.state_dict())
                    critic_target.load_state_dict(critic_ddpg.state_dict())
                    step_count=0

        # Close environment
        env.close()

        # Save policy
        # torch.save(actor_ddpg.state_dict(), "DDPG\\pendulum_actor.pt")
        # torch.save(critic_ddpg.state_dict(), "DDPG\\pendulum_critic.pt")

        # # Create new graph 
        # plt.figure(1)

        # # Plot average rewards (Y-axis) vs episodes (X-axis)
        # sum_rewards = np.zeros(episodes)
        # for x in range(episodes):
        #     sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        # plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.plot(sum_rewards)
        
        # # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        # plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.plot(epsilon_history)
        
        # # Save plots
        # plt.savefig("DDPG\\cartpole_dql.png")

    # Optimize policy network
    def optimize(self, mini_batch, actor_ddpg,actor_target, critic_ddpg,critic_target):

        current_q_list = []
        target_q_list = []
        actor_list = []
        # actor_loss = torch.tensor([0])

        for state, action, new_state, reward, terminated in mini_batch:
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            new_state = torch.FloatTensor(new_state)
            reward = torch.FloatTensor([reward])

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target_q = reward
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target_q = reward + self.discount_factor_g * critic_target(new_state,actor_target(new_state))

            # Get the current set of Q values
            current_q = critic_ddpg(state,action)
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q_list.append(target_q)

            # Get actor loss
            actor_loss_minibatch = -critic_ddpg(state, actor_ddpg(state))
            actor_list.append(actor_loss_minibatch)
            # actor_loss = actor_loss + actor_loss_minibatch
                
        # Compute loss for the whole minibatch
        critic_loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        actor_loss = torch.mean(torch.stack(actor_list))
        # actor_loss = actor_loss/len(mini_batch)
        # return critic_loss/len(mini_batch)
        

        # Optimize critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Optimize actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()


    def select_action(self, actor_net,state):
        state = torch.FloatTensor(state)
        action = actor_net.forward(state)
        return action.data.numpy()

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

        actor_ddpg.load_state_dict(torch.load("DDPG\\pendulum_actor.pt"))
        actor_ddpg.eval()    # switch model to evaluation mode

        critic_ddpg.load_state_dict(torch.load("DDPG\\pendulum_critic.pt"))
        critic_ddpg.eval()    # switch model to evaluation mode

        state = env.reset()[0]
        for _ in range(episodes):
            env.render()  
            with torch.no_grad():
                action = actor_ddpg(state)
            state, reward, done, _, _ = env.step(action)
            if done:
            # state, _ = env.reset()
                print("end!")
                break

        env.close()




if __name__ == '__main__':
    # env = gym.make('Pendulum-v1')
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # max_action = env.action_space.high[0]
    # min_action = env.action_space.low[0]
    # actor = Actor(state_dim,action_dim,max_action)
    # critic = Critic(state_dim,action_dim)
    # state, _ = env.reset()
    # action = np.array([env.action_space.sample()])
    # output = actor.forward(torch.FloatTensor(state))
    # s = torch.FloatTensor(state)
    # a = torch.FloatTensor(action).squeeze(1)
    # x = x = torch.cat((s, a))
    # out = critic.forward(s,output)
    # print(out)
    pendulum = DDPG()
    pendulum.train(100)


