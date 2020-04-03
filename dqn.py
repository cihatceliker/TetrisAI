import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math
import sys
import pickle

device = torch.device("cuda")


class DuelingDeepQNetwork(nn.Module):

    def __init__(self, n_actions):
        super(DuelingDeepQNetwork, self).__init__()
        self.conv = CNN(n_actions)
        self.fc1 = nn.Linear(1863, 256)
        self.fc2 = nn.Linear(256, 256)
        self.V = nn.Linear(256, 1)
        self.A = nn.Linear(256, n_actions)
        self.to(device)

    def forward(self, state, next_piece):
        features = self.conv(state)
        x = torch.relu(self.fc1(torch.cat([features, next_piece], dim=2)))
        x = torch.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)
        return V, A


class CNN(nn.Module):

    def __init__(self, out_size):
        super(CNN, self).__init__()
        in_channels = 8
        def create_block(in_, out_, sz):
            return nn.Sequential(
                nn.Conv2d(in_, out_, sz),
                nn.MaxPool2d(2),
                nn.ReLU()
            )
        self.conv_row = nn.Conv2d(in_channels, 16, (1, 10))
        self.conv_col = nn.Conv2d(in_channels, 16, (20, 1))
        self.conv_initial = nn.Conv2d(in_channels, 16, 5, padding=2)
        self.block1 = create_block(16, 24, 3)
        self.block2 = create_block(24, 128, (2,3))

    def forward(self, state):
        row = torch.relu(self.conv_row(state)).view(state.size(0), 1, -1)
        col = torch.relu(self.conv_col(state)).view(state.size(0), 1, -1)
        x = torch.relu(self.conv_initial(state))
        x_skip = self.block1(x)
        x = self.block2(x_skip)
        return torch.cat([
            row, col,
            x_skip.view(state.size(0), 1, -1),
            x.view(state.size(0), 1, -1)
        ], dim=2)


class Agent():
    
    def __init__(self, num_actions, eps_start=1.0, eps_end=0.05, eps_decay=25e-4, tau=1e-3,
                            gamma=0.999, memory_capacity=5000, batch_size=128, alpha=5e-4):
        self.local_Q = DuelingDeepQNetwork(num_actions)
        self.target_Q = DuelingDeepQNetwork(num_actions)
        self.target_Q.load_state_dict(self.local_Q.state_dict())
        self.target_Q.eval()
        self.optimizer = optim.Adam(self.local_Q.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(memory_capacity)
        self.indexes = np.arange(self.batch_size)
        self.scores = []
        self.episodes = []
        self.durations = []
        self.start = 1

    def select_action(self, state, next_piece):
        if np.random.random() > self.eps_start:
            self.local_Q.eval()
            with torch.no_grad():
                obs = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
                next_piece = torch.tensor(next_piece, device=device, dtype=torch.float).view(1,1,7)
                action = torch.argmax(self.local_Q(obs, next_piece)[1]).item()
            self.local_Q.train()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def learn(self):
        if self.batch_size >= len(self.replay_memory.memory):
            return

        state_batch, next_piece_batch, action_batch, reward_batch, next_state_batch, next_next_piece_batch, done_batch = \
                self.replay_memory.sample(self.batch_size)

        local_V, local_A = self.local_Q(state_batch, next_piece_batch)        
        local_next_V, local_next_A = self.local_Q(next_state_batch, next_next_piece_batch)        
        target_next_V, target_next_A = self.target_Q(next_state_batch, next_next_piece_batch)        
        
        pred = (local_V + (local_A - local_A.mean(dim=2, keepdim=True)))[self.indexes, 0, action_batch]
        nx = (target_next_V + (target_next_A - target_next_A.mean(dim=2, keepdim=True)))
        ev = (local_next_V + (local_next_A - local_next_A.mean(dim=2, keepdim=True)))

        target_out = torch.argmax(ev, dim=2).squeeze(1)
        target = reward_batch + self.gamma * nx[self.indexes,0,target_out] * done_batch

        self.optimizer.zero_grad()
        loss = self.loss(pred, target).to(device)
        loss.backward()
        self.optimizer.step()
        
        for target_param, local_param in zip(self.target_Q.parameters(), self.local_Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

        self.eps_start = self.eps_start - self.eps_decay if self.eps_start > self.eps_end else self.eps_end

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
                reward = self.memory[int(self.position)][2]
                rnd = np.random.random()
                #if reward > 0 or (reward != 0 and rnd < 0.9):
                if reward > 1 and rnd < 0.95:
                    self.position = (self.position + 1) % self.capacity
                    print("skipped")
                    self.push(args)
                    return
            """
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
