import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class DeepQNetwork(nn.Module):
    def _init_(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self)._init_()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(input_dims[0], self.fc1_dims)  
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent:
    def _init_(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size, eps_end, eps_dec, target_update, save_dir='checkpoints'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.target_update = target_update
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.Q_eval = DeepQNetwork(lr, input_dims=input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions)
        self.Q_target = DeepQNetwork(lr, input_dims=input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions)
        self.Q_target.load_state_dict(self.Q_eval.state_dict()) 

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64) 
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.load_checkpoint()

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        observation = np.array(observation, dtype=np.float32)  
        observation = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval(observation)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = T.tensor(self.state_memory[batch], dtype=T.float32).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch], dtype=T.float32).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch], dtype=T.float32).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.bool).to(self.Q_eval.device)
        action_batch = T.tensor(self.action_memory[batch], dtype=T.int64).to(self.Q_eval.device)

        q_eval = self.Q_eval(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        q_next = self.Q_target(new_state_batch).detach()
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval)
        loss.backward()
        self.Q_eval.optimizer.step()

        if self.mem_cntr % self.target_update == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict()) 

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_checkpoint(self):
        checkpoint = {
            'Q_eval_state_dict': self.Q_eval.state_dict(),
            'Q_target_state_dict': self.Q_target.state_dict(),
            'optimizer_state_dict': self.Q_eval.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'mem_cntr': self.mem_cntr,
            'state_memory': self.state_memory,
            'new_state_memory': self.new_state_memory,
            'action_memory': self.action_memory,
            'reward_memory': self.reward_memory,
            'terminal_memory': self.terminal_memory,
        }
        T.save(checkpoint, os.path.join(self.save_dir, 'checkpoint.pth'))

    def load_checkpoint(self):
        if os.path.isfile(os.path.join(self.save_dir, 'checkpoint.pth')):
            checkpoint = T.load(os.path.join(self.save_dir, 'checkpoint.pth'))
            self.Q_eval.load_state_dict(checkpoint['Q_eval_state_dict'])
            self.Q_target.load_state_dict(checkpoint['Q_target_state_dict'])
            self.Q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.mem_cntr = checkpoint['mem_cntr']
            self.state_memory = checkpoint['state_memory']
            self.new_state_memory = checkpoint['new_state_memory']
            self.action_memory = checkpoint['action_memory']
            self.reward_memory = checkpoint['reward_memory']
            self.terminal_memory = checkpoint['terminal_memory']