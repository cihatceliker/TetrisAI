import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math
import sys

device = torch.device("cuda")

class Brain(nn.Module):

    def __init__(self, out_size):
        super(Brain, self).__init__()
        self.out_size = out_size
        self.cnn = nn.Sequential(*[
            nn.Conv2d(1, 8, kernel_size=4), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4), nn.ReLU()
        ])
        self.rnn = nn.RNN(32*11, 256, 1, batch_first=True)
        self.out = nn.Linear(256, out_size)

    def init_hidden(self, sz=256):
        return torch.zeros((sz), device=device, dtype=torch.float).view(1,1,sz)

    def forward(self, x, hidden):
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return self.out(torch.relu(output)), hidden
        
    def extract_features(self, stacked_frames):
        timesteps = len(stacked_frames)
        cnn_in = torch.tensor(stacked_frames, device=device, dtype=torch.float)
        cnn_out = self.cnn(cnn_in.view(timesteps, 1, 20, 10)).view(1, timesteps, -1)
        return cnn_out

    def select_action(self, stacked_state):
        self.rnn.flatten_parameters()
        x = self.extract_features(stacked_state)
        hidden = self.init_hidden()
        for i in range(x.size(1)):
            output, hidden = self.rnn(x[:,i].unsqueeze(0), hidden)
        return self.out(output)
    

class Agent():
    
    def __init__(self, num_actions, eps_start=1.0, eps_end=0.05,
                 eps_decay=0.998, gamma=0.99, alpha=5e-2, memory_capacity=3e3, tau=1e-3):
        self.local_Q = Brain(num_actions).to(device)
        self.target_Q = Brain(num_actions).to(device)
        self.target_Q.load_state_dict(self.local_Q.state_dict())
        self.target_Q.eval()
        self.optimizer = optim.Adam(self.local_Q.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        #self.loss = nn.SmoothL1Loss()
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.gamma = gamma
        self.replay_memory = ReplayMemory(memory_capacity)
        self.scores = []
        self.episodes = []
        self.durations = []
        self.start = 1

    def store_experience(self, *args):
        self.replay_memory.push(args)

    def select_action(self, stacked_state):
        if np.random.random() > self.eps_start:
            with torch.no_grad():
                out = self.local_Q.select_action(stacked_state)
                action = torch.argmax(out).item()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def learn(self, batch=None):
        for _ in range(4):
            stacked_states, stacked_actions, reward, done = batch if batch else self.replay_memory.sample()

            cnn_out = self.local_Q.extract_features(stacked_states)
            loss = 0
            hidden = self.local_Q.init_hidden()

            for timestep in range(len(stacked_actions)):
                action = stacked_actions[timestep]

                state_features = cnn_out[:,timestep].unsqueeze(1)
                next_state_features = cnn_out[:,timestep+1].unsqueeze(1)
                
                output, hidden_ = self.local_Q(state_features, hidden)
                target = output.clone()

                next_out, _ = self.target_Q(next_state_features, hidden)
                hidden = hidden_
                next_out = torch.max(next_out, dim=2)[0].squeeze(1)

                # vanilla dqn
                target[:,:,action] = reward + self.gamma * next_out * done

                loss += self.loss(output, target.detach()).to(device)


            self.optimizer.zero_grad()
            loss.backward()
            #for param in self.local_Q.parameters(): param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            # soft update
            for target_param, local_param in zip(self.target_Q.parameters(), self.local_Q.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        else:
            while self.memory[int(self.position)][2] > 1:
                self.position = (self.position + 1) % self.capacity
                #print("skipped")
            while self.memory[int(self.position)][2] == 1 and np.random.random() < 0.95:
                self.position = (self.position + 1) % self.capacity

        self.memory[int(self.position)] = args[0]
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.choice(self.memory)