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
        self.conv0R = nn.Conv2d(6, 32, kernel_size=(1,10))
        self.conv0C = nn.Conv2d(6, 32, kernel_size=(20,1))
        self.conv1_1 = nn.Conv2d(6, 32, kernel_size=5, padding=2)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(32, 32, kernel_size=1)
        self.conv1_4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv1R = nn.Conv2d(32, 32, kernel_size=(1,5))
        self.conv1C = nn.Conv2d(32, 32, kernel_size=(10,1))
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv2_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2R = nn.Conv2d(64, 32, kernel_size=(1,2))
        self.conv2C = nn.Conv2d(64, 32, kernel_size=(5,1))
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv3_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(4167, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, out_size)


    def forward(self, state, next_piece):
        xR = torch.relu(self.conv0R(state)).view(state.size(0), 1, -1)
        xC = torch.relu(self.conv0C(state)).view(state.size(0), 1, -1)
        x1 = torch.relu(self.conv1_1(state))
        x1 = torch.relu(self.conv1_2(x1))
        x1 = torch.relu(self.conv1_3(x1))
        x1 = torch.relu(self.conv1_4(x1))
        x1 = torch.max_pool2d(x1, 2)
        x1R = torch.relu(self.conv1R(x1)).view(state.size(0), 1, -1)
        x1C = torch.relu(self.conv1C(x1)).view(state.size(0), 1, -1)
        x2 = torch.relu(self.conv2_1(x1))
        x2 = torch.relu(self.conv2_2(x2))
        x2 = torch.relu(self.conv2_3(x2))
        x2 = torch.max_pool2d(x2, 2)
        x2R = torch.relu(self.conv2R(x2)).view(state.size(0), 1, -1)
        x2C = torch.relu(self.conv2C(x2)).view(state.size(0), 1, -1)
        x3 = torch.relu(self.conv3_1(x2))
        x3 = torch.relu(self.conv3_2(x3))
        x3 = torch.relu(self.conv3_3(x3))
        x3 = torch.max_pool2d(x3, 2)
        x = torch.cat([
            xR, xC, x1R, x1C, x2R, x2C,
            x1.view(state.size(0), 1, -1),
            x2.view(state.size(0), 1, -1),
            x3.view(state.size(0), 1, -1),
            next_piece
        ], dim=2)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class Agent():
    
    def __init__(self, num_actions, eps_start=1.0, eps_end=0.05, eps_decay=0.996,
                            gamma=0.99, memory_capacity=20000, batch_size=256, alpha=5e-4, tau=1e-3):
        self.local_Q = Brain(num_actions).to(device)
        self.target_Q = Brain(num_actions).to(device)
        self.target_Q.load_state_dict(self.local_Q.state_dict())
        self.target_Q.eval()
        self.optimizer = optim.Adam(self.local_Q.parameters(), lr=alpha)
        #self.loss = nn.SmoothL1Loss()
        self.loss = nn.MSELoss()
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

    def select_action(self, state, next_piece):
        if np.random.random() > self.eps_start:
            #self.local_Q.eval()
            with torch.no_grad():
                obs = torch.tensor(state, device=device, dtype=torch.float).view(1,6,20,10)
                next_piece = torch.tensor(next_piece, device=device, dtype=torch.float).view(1,1,7)
                action = torch.argmax(self.local_Q(obs, next_piece)).item()
            #self.local_Q.train()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def learn(self):
        if self.batch_size >= len(self.replay_memory.memory):
            return
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, next_piece_batch = \
                self.replay_memory.sample(self.batch_size)

        local_out = self.local_Q(state_batch, next_piece_batch)

        target = local_out.clone()
        target_out = torch.max(self.target_Q(next_state_batch, next_piece_batch), dim=2)[0].squeeze(1)

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
            if self.position == 0:
                self.position = 4000
            #if self.memory[int(self.position)][2] > 0 or \
            if self.memory[int(self.position)][2] > 8 or \
                (self.memory[int(self.position)][2] < -30 and np.random.random() < 0.5):# or \
                #(self.memory[int(self.position)][2] > 0 and np.random.random() < 0.5):

                self.position = (self.position + 1) % self.capacity
                self.push(args)
                return

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
        next_piece_batch = torch.tensor(batch[5], device=device, dtype=torch.float).unsqueeze(1)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, next_piece_batch