import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math
import pickle
import sys

device = torch.device("cuda")


class Network(nn.Module):

    def __init__(self, n_actions):
        super(Network, self).__init__()
        self.conv = CNN(n_actions)
        self.fc1 = nn.Linear(1447, 256)
        self.out = nn.Linear(256, n_actions)
        self.to(device)

    def forward(self, state, next_piece):
        features = self.conv(state)
        x = torch.relu(self.fc1(torch.cat([features, next_piece], dim=2)))
        return self.out(x)


class CNN(nn.Module):

    def __init__(self, out_size):
        super(CNN, self).__init__()
        in_channels = 8
        self.convR = nn.Conv2d(in_channels, 8, (20,1))
        self.convC = nn.Conv2d(in_channels, 8, (1,10))
        self.conv1 = nn.Conv2d(in_channels, 16, 5, padding=2)
        self.block1 = nn.Sequential(
            nn.Conv2d(16, 24, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

    def forward(self, state):
        xR = torch.relu(self.convR(state)).view(state.size(0), 1, -1)
        xC = torch.relu(self.convC(state)).view(state.size(0), 1, -1)
        x = torch.relu(self.conv1(state))
        x = self.block1(x).view(state.size(0), 1, -1)
        return torch.cat([xR, xC, x], dim=2)


class Agent():
    
    def __init__(self, num_actions, eps_start=1.0, eps_end=0, eps_decay=0.996,
                            gamma=0.992, memory_capacity=20000, batch_size=128, alpha=10e-4, tau=1e-3):
        self.local_Q = Network(num_actions).to(device)
        self.target_Q = Network(num_actions).to(device)
        self.target_Q.load_state_dict(self.local_Q.state_dict())
        self.target_Q.eval()
        self.optimizer = optim.Adam(self.local_Q.parameters(), lr=alpha)
        self.loss = nn.SmoothL1Loss()
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(memory_capacity)
        self.indexes = np.arange(self.batch_size)
        self.scores = []
        self.episodes = []
        self.durations = []
        self.start = 1
    
    def store_experience(self, *args):
        self.replay_memory.push(args)

    def select_action(self, state, next_piece):
        if np.random.random() > self.eps_start:
            #self.local_Q.eval()
            with torch.no_grad():
                obs = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
                next_piece = torch.tensor(next_piece, device=device, dtype=torch.float).view(1,1,7)
                action = torch.argmax(self.local_Q(obs, next_piece)).item()
            #self.local_Q.train()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def learn(self):
        ln = len(self.replay_memory.memory)
        if self.batch_size >= ln or ln < self.replay_memory.capacity:
            return
        
        state_batch, next_piece_batch, action_batch, reward_batch, next_state_batch, next_next_piece_batch, done_batch = \
            self.replay_memory.sample(self.batch_size)
        
        max_actions = torch.argmax(self.local_Q(next_state_batch, next_next_piece_batch), dim=2).squeeze(1)
        prediction = self.local_Q(state_batch, next_piece_batch)[self.indexes,0,action_batch]

        with torch.no_grad():
            evaluated = self.target_Q(next_state_batch, next_next_piece_batch)[self.indexes,0,max_actions]
            evaluated = reward_batch + self.gamma * evaluated * done_batch

        self.optimizer.zero_grad()
        self.loss(prediction, evaluated).to(device).backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_Q.parameters(), self.local_Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
        
        self.eps_start = max(self.eps_end, self.eps_decay * self.eps_start)
        

    def save(self, filename):
        pickle_out = open(filename+".tt","wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()


def load_agent(filename):
    pickle_in = open(filename, mode="rb")
    agent = pickle.load(pickle_in)
    pickle_in.close()
    return agent

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        batch = list(zip(*batch))

        state_batch = torch.tensor(batch[0], device=device, dtype=torch.float)
        next_piece_batch = torch.tensor(batch[1], device=device, dtype=torch.float).unsqueeze(1)
        action_batch = torch.tensor(batch[2], device=device)
        reward_batch = torch.tensor(batch[3], device=device)
        next_state_batch = torch.tensor(batch[4], device=device, dtype=torch.float)
        next_next_piece_batch = torch.tensor(batch[5], device=device, dtype=torch.float).unsqueeze(1)
        done_batch = torch.tensor(batch[6], device=device)

        return state_batch, next_piece_batch, action_batch, reward_batch, next_state_batch, next_next_piece_batch, done_batch

