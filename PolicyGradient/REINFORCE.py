import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define model for approximate policy
class policy(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = F.softmax(self.out(x), dim=-1) # Calculate output
        return x
    
    
# FrozeLake REINFORCE
class REINFORCE():
    def __init__(self):
        # Hyperparameters (adjustable)
        self.learning_rate_a = 0.001         # learning rate (alpha)
        self.discount_factor = 0.9         # discount rate (gamma)   
        self.max_t = 1000   # time for collect  
        self.network_print = 100          # print
        self.optimizer = None                # NN Optimizer. Initialize later.

        self.ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    # Train the FrozeLake environment
    def train(self, episodes, render=False, is_slippery=False):
        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        

        # Create policy network. Number of nodes in the hidden layer can be adjusted.
        policy_reinforce = policy(in_states=num_states, h1_nodes=64, out_actions=num_actions)


        # print('Policy (random, before training):')
        # self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_reinforce.parameters(), lr=self.learning_rate_a)

        scores = []
        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            saved_log_probs = []
            rewards = []    

            for t in range(self.max_t):
                action, log_prob = self.select_action(policy_reinforce,state,num_states)
                saved_log_probs.append(log_prob)
                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)
                rewards.append(reward)
                if (terminated or truncated):
                    break

            # Calculate total expected reward
            scores.append(sum(rewards))

            # Recalculate the total reward applying discounted factor
            discounts = [self.discount_factor ** i for i in range(len(rewards) + 1)]
            R = sum([a * b for a,b in zip(discounts, rewards)])

            # Calculate the loss 
            # Note: L(theta) = -R_t*sigma_{t=0}^{T}(log(pi_theta(a_t|s_t)))
            policy_loss = []
            for log_prob in saved_log_probs:
                # Note that we are using Gradient Ascent. 
                policy_loss.append(-log_prob * R)
            # After that, we concatenate whole policy loss in 0th dimension
            policy_loss = torch.stack(policy_loss).sum()
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()



        # Close environment
        env.close()

        # Save policy
        torch.save(policy_reinforce.state_dict(), "PolicyGradient\\frozen_lake_REINFORCE.pt")

        # Create new graph 
        plt.figure(1)

        # # Plot average rewards (Y-axis) vs episodes (X-axis)
        # sum_rewards = np.zeros(episodes)
        # for x in range(episodes):
        #     sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        # plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.plot(sum_rewards)
        
        # # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        # plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.plot(epsilon_history)
        plt.plot(np.arange(1, len(scores)+1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        
        # Save plots
        plt.savefig("PolicyGradient\\frozen_lake_REINFORCE.png")




    def select_action(self,policy,state,num_states):
        state = self.state_to_input(state, num_states)
        probs = policy.forward(state)
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)

    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, is_slippery=False):
        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_reinforce = policy(in_states=num_states, h1_nodes=64, out_actions=num_actions)
        policy_reinforce.load_state_dict(torch.load("PolicyGradient\\frozen_lake_REINFORCE.pt"))
        policy_reinforce.eval()    # switch model to evaluation mode


        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                state = self.state_to_input(state, num_states)

                with torch.no_grad():
                    probs = policy_reinforce.forward(state) 
                    action = torch.argmax(probs).item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)

        env.close()



if __name__ == '__main__':
    # frozen_lake = DQN()
    # is_slippery = False
    # # frozen_lake.train(1000, is_slippery=is_slippery)
    # frozen_lake.test(1, is_slippery=is_slippery)
    frozen_lake = REINFORCE()
    is_slippery = False
    # frozen_lake.train(10000, is_slippery=is_slippery)
    frozen_lake.test(1, is_slippery=is_slippery)
