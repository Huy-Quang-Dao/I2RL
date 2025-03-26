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
    def __init__(self, in_states, hPolicy_nodes,VPolicy_nodes, out_actions):
        super().__init__()

        # Define action network layers
        self.policy_fc1 = nn.Linear(in_states, hPolicy_nodes)   # first fully connected layer
        self.policy_fc2 = nn.Linear(hPolicy_nodes, out_actions) # ouptut layer w

        # Define value network layers
        self.value_fc1 = nn.Linear(in_states, VPolicy_nodes)   # first fully connected layer
        self.value_fc2 = nn.Linear(VPolicy_nodes, 1) # ouptut layer w

    def forward(self, x):
        # action network
        action_probs = F.relu(self.policy_fc1(x)) # Apply rectified linear unit (ReLU) activation
        action_probs = F.softmax(self.policy_fc2(action_probs), dim=-1) # Calculate output

        # value network
        state_value = F.relu(self.value_fc1(x))
        state_value = self.value_fc2(state_value)

        return action_probs,state_value
    
    
# FrozeLake REINFORCE
class REINFORCE():
    def __init__(self):
        # Hyperparameters (adjustable)
        self.learning_rate_a = 0.01         # learning rate (alpha)
        self.discount_factor = 0.9         # discount rate (gamma)   
        self.max_t = 1000   # time for collect  
        self.network_print = 100          # print
        self.optimizer = None                # NN Optimizer. Initialize later.
        self.loss_fn = nn.MSELoss()          # Value NN Loss function.

        self.ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    # Train the FrozeLake environment
    def train(self, episodes, render=False, is_slippery=False):
        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        

        # Create policy network. Number of nodes in the hidden layer can be adjusted.
        policy_reinforce = policy(in_states=num_states,hPolicy_nodes=64,VPolicy_nodes=128, out_actions=num_actions)


        # print('Policy (random, before training):')
        # self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_reinforce.parameters(), lr=self.learning_rate_a)

        scores = []
        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            saved_log_probs = []
            rewards = []  
            state_values = []  

            for t in range(self.max_t):
                action, log_prob,state_value = self.set_action(policy_reinforce,state,num_states)
                saved_log_probs.append(log_prob)
                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)
                rewards.append(reward)
                state_values.append(state_value)
                if (terminated or truncated):
                    break

            # Calculate total expected reward
            scores.append(sum(rewards))

            # # Recalculate the total reward applying discounted factor
            # discounts = [self.discount_factor ** i for i in range(len(rewards) + 1)]
            # G = [a * b for a,b in zip(discounts, rewards)]
            Re = 0
            G = []
            for r in reversed(rewards):
                Re = r + self.discount_factor * Re
                G.insert(0, Re)

            G = torch.tensor(G)
            # G = (G - G.mean()) / (G.std() + 1e-9)

            # Calculate the loss 
            # Note: L(theta) = -R_t*sigma_{t=0}^{T}(log(pi_theta(a_t|s_t)))
            policy_loss = []
            value_loss = []
            for log_prob_t,state_value_t,R_t in zip(saved_log_probs,state_values,G):
                # Note that we are using Gradient Ascent. 
                # return print(R_t)
                policy_loss.append(-log_prob_t * (R_t-state_value_t.item()))
                value_loss.append(self.loss_fn(state_value_t, torch.tensor([R_t])))

            # After that, we concatenate whole policy loss in 0th dimension
            policy_loss = torch.stack(policy_loss).sum()
            value_loss = torch.stack(value_loss).sum()
            network_loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            network_loss.backward()
            self.optimizer.step()



        # Close environment
        env.close()

        # Save policy
        torch.save(policy_reinforce.state_dict(), "PolicyGradient\\frozen_lake_REINFORCEbaseline.pt")

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
        plt.savefig("PolicyGradient\\frozen_lake_REINFORCEbaseline.png")




    def set_action(self,policy,state,num_states):
        state = self.state_to_input(state, num_states)
        probs,state_value = policy.forward(state)
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action),state_value

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
        policy_reinforce = policy(in_states=num_states,hPolicy_nodes=64,VPolicy_nodes=128, out_actions=num_actions)
        policy_reinforce.load_state_dict(torch.load("PolicyGradient\\frozen_lake_REINFORCEbaseline.pt"))
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
                    probs,_ = policy_reinforce.forward(state) 
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
