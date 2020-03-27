import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math
import sys
import pickle

device = torch.device("cuda")

class Brain(nn.Module):

    def __init__(self, out_size):
        super(Brain, self).__init__()
        self.out_size = out_size
        self.convR = nn.Conv2d(6, 2*8, kernel_size=(1,10))
        self.convC = nn.Conv2d(6, 2*8, kernel_size=(20,1))
        self.conv1 = nn.Conv2d(6, 2*16, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4)
        self.fc = nn.Linear(480+64*11, 256)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(256, out_size)

    def forward(self, state):
        xR = torch.relu(self.convR(state)).view(state.size(0), 1, -1)
        xC = torch.relu(self.convC(state)).view(state.size(0), 1, -1)
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x)).view(state.size(0), 1, -1)
        x = torch.cat([x, xR, xC], dim=2)
        x = self.drop(self.fc(x))
        return self.out(torch.relu(x))


class Agent():
    
    def __init__(self, num_actions, eps_start=1.0, eps_end=0.03, eps_decay=0.996,
                            gamma=0.99, memory_capacity=20000, batch_size=256, alpha=5e-4, tau=1e-3):
        self.local_Q = Brain(num_actions).to(device)
        self.target_Q = Brain(num_actions).to(device)
        self.target_Q.load_state_dict(self.local_Q.state_dict())
        self.target_Q.eval()
        self.optimizer = optim.Adam(self.local_Q.parameters(), lr=alpha)
        self.loss = nn.SmoothL1Loss()
        #self.loss = nn.MSELoss()
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
        self.start = 0

    def select_action(self, state):
        if np.random.random() > self.eps_start:
            self.local_Q.eval()
            with torch.no_grad():
                obs = torch.tensor(state, device=device, dtype=torch.float).view(1,6,20,10)
                action = torch.argmax(self.local_Q(obs)).item()
            self.local_Q.train()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def learn(self):
        if self.batch_size >= len(self.replay_memory.memory):
            return
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
                self.replay_memory.sample(self.batch_size)
        local_out = self.local_Q(state_batch)

        target = local_out.clone()
        target_out = torch.max(self.target_Q(next_state_batch), dim=2)[0].squeeze(1)

        target[self.indexes,0,action_batch] = reward_batch + self.gamma * target_out * done_batch
        
        loss = self.loss(local_out, target.detach()).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_Q.parameters(), self.local_Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        pickle_out = open(filename+".tt","wb")
        saved = {
            "local": self.local_Q.state_dict(),
            "target": self.target_Q.state_dict(),
            "optimizer": self.optimizer,
            "loss": self.loss,
            "eps_decay": self.eps_decay,
            "eps_end": self.eps_end,
            "eps_start": self.eps_start,
            "tau": self.tau,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "scores": self.scores,
            "episodes": self.episodes,
            "durations": self.durations,
            "start": self.start,
            "replay_memory": self.replay_memory,
        }
        pickle.dump(saved, pickle_out)
        pickle_out.close()

    def load(self, filename):
        pickle_in = open(filename+".tt", mode="rb")
        info = pickle.load(pickle_in)
        self.local_Q.load_state_dict(info["local"])
        self.target_Q.load_state_dict(info["target"])
        self.optimizer = info["optimizer"]
        self.loss = info["loss"]
        self.eps_decay = info["eps_decay"]
        self.eps_end = info["eps_end"]
        self.eps_start = info["eps_start"]
        self.tau = info["tau"]
        self.gamma = info["gamma"]
        self.batch_size = info["batch_size"]
        self.scores = info["scores"]
        self.episodes = info["episodes"]
        self.durations = info["durations"]
        self.start = info["start"]
        self.replay_memory = info["replay_memory"]
        pickle_in.close()
         
    def store_experience(self, *args):
        self.replay_memory.push(args)


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        else:
            """
            if self.memory[int(self.position)][2] > 0 and np.random.random() < 1:#0.9
                self.position = (self.position + 1) % self.capacity
                self.push(args)
                return
            """
        self.memory[int(self.position)] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        batch = list(zip(*batch))

        state_batch = torch.tensor(batch[0], device=device, dtype=torch.float)
        action_batch = torch.tensor(batch[1], device=device)
        reward_batch = torch.tensor(batch[2], device=device)
        next_state_batch = torch.tensor(batch[3], device=device, dtype=torch.float)
        done_batch = torch.tensor(batch[4], device=device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch