import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math
import sys

device = torch.device("cuda")

class Brain(nn.Module):

    def __init__(self, in_, out_size):
        super(Brain, self).__init__()
        self.in_size = in_
        self.out_size = out_size
        self.conv1 = nn.Conv2d(in_, 32, kernel_size=(11,7), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(11,5))
        self.fc = nn.Linear(256, 128)
        self.out = nn.Linear(128, out_size)
        
    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = x.view(state.size(0), 1, -1)
        x = torch.relu(self.fc(x))
        return self.out(x)


class Agent():
    
    def __init__(self, local_Q, target_Q, num_actions, eps_start=1.0, eps_end=0.1,
                 eps_decay=0.996, gamma=0.99, alpha=5e-4, batch_size=128, memory_capacity=2e5, tau=1e-3):
        self.local_Q = local_Q.to(device)
        self.target_Q = target_Q.to(device)
        self.target_Q.load_state_dict(self.local_Q.state_dict())
        self.target_Q.eval()
        self.optimizer = optim.Adam(self.local_Q.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(memory_capacity)
        self.scores = []
        self.episodes = []
        self.batch_index = np.arange(self.batch_size)

    def store_experience(self, *args):
        self.replay_memory.push(args)

    def select_action(self, state):
        sample = np.random.random()
        if sample > self.eps_start:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
                action = torch.argmax(self.local_Q(state)).item()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def learn(self):
        if len(self.replay_memory.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_memory.sample(self.batch_size)

        output = self.local_Q(state_batch).squeeze(1)
        target = output.clone()

        # vanilla dqn
        target[self.batch_index, action_batch] = reward_batch + \
                self.gamma * torch.max(self.target_Q(next_state_batch), dim=2)[0].squeeze(1) * done_batch

        loss = self.loss(output, target.detach()).to(device)

        self.optimizer.zero_grad()
        loss.backward()
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
        elif self.memory[int(self.position)][2] > 0:
            self.position = (self.position + 1) % self.capacity
            print("skipped")

        self.memory[int(self.position)] = args[0]
        self.position = (self.position + 1) % self.capacity

    def sample(self, size):
        batch = random.sample(self.memory, size)

        batch = list(zip(*batch))
        
        state_batch = torch.tensor(batch[0], device=device, dtype=torch.float)
        action_batch = torch.tensor(batch[1])
        reward_batch = torch.tensor(batch[2], device=device)
        next_state_batch = torch.tensor(batch[3], device=device, dtype=torch.float)
        done_batch = torch.tensor(batch[4], device=device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch