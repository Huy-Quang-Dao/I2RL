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

# Define model
class Net(nn.Module):
    def __init__(self, in_states, h1_nodes,h2_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(h1_nodes,h2_nodes)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(h2_nodes, out_actions) # ouptut layer w
        self.out.weight.data.normal_(0,0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))
        x = self.out(x)         # Calculate output
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
    
# FrozeLake Deep Q-Learning
class DQN():
    def __init__(self):
        super(DQN, self).__init__()
        # Hyperparameters (adjustable)
        self.learning_rate_a = 0.01         # learning rate (alpha)
        self.discount_factor_g = 0.9         # discount rate (gamma)    
        self.network_sync_rate = 100          # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = 2000       # size of replay memory
        self.mini_batch_size = 32           # size of the training data set sampled from the replay memory

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        self.ACTIONS = ['L','R']     # for printing 0,1 => L(eft),R(ight)



    # Train the FrozeLake environment
    def train(self, episodes, render=False):
        # Create FrozenLake instance
        env = gym.make('CartPole-v1', render_mode='human' if render else None)
        # env = env.unwrapped
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = Net(in_states=num_states, h1_nodes=64,h2_nodes=32, out_actions=num_actions)
        target_dqn = Net(in_states=num_states, h1_nodes=64,h2_nodes=32, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
            
        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=right
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode)>0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon = epsilon
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        # Close environment
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "DQN\\cartpole_dql.pt")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig("DQN\\cartpole_dql.png")

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        return torch.tensor(state, dtype=torch.float32).squeeze()

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes,render=True):
        # Create FrozenLake instance
        env = gym.make('CartPole-v1', render_mode='human' if render else None)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = Net(in_states=num_states, h1_nodes=64,h2_nodes=32, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("DQN\\cartpole_dql.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        state = env.reset()[0]
        for _ in range(episodes):
            env.render()  
            with torch.no_grad():
                action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
            state, reward, done, _, _ = env.step(action)
            if done:
            # state, _ = env.reset()
                print("end!")
                break

        env.close()



if __name__ == '__main__':
    cartpole = DQN()
    # cartpole.train(1000)
    cartpole.test(200)

