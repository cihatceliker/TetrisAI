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
        self.convR = nn.Conv2d(1, 4, kernel_size=(1,10))
        self.convC = nn.Conv2d(1, 4, kernel_size=(20,1))
        self.conv1 = nn.Conv2d(1, 6, kernel_size=4)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=4)
        self.conv3 = nn.Conv2d(12, 16, kernel_size=4)
        self.rnn = nn.GRU(16*11+120, 64, 1)
        self.out = nn.Linear(64, out_size)

    def init_hidden(self, sz=64):
        return torch.zeros((sz), device=device, dtype=torch.float).view(1,1,sz).detach()

    def forward(self, batch):
        self.rnn.flatten_parameters()
        cnn_out = self.extract_features(batch)
        output, hidden = self.rnn(cnn_out)
        return self.out(output), hidden
        
    def extract_features(self, state):
        xR = torch.relu(self.convR(state)).view(state.size(0), 1, -1)
        xC = torch.relu(self.convC(state)).view(state.size(0), 1, -1)
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x)).view(state.size(0), 1, -1)
        return torch.cat([x, xR, xC], dim=2)

    def select_action(self, state, hidden):
        self.rnn.flatten_parameters()
        cnn_out = self.extract_features(state)
        output, hidden = self.rnn(cnn_out, hidden)
        return self.out(output), hidden
    

class Agent():
    
    def __init__(self, num_actions, eps_start=1.0, eps_end=0.1, eps_decay=0.995,
                            gamma=0.99, memory_capacity=500, alpha=1e-3, tau=1e-3):
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
        self.replay_memory = ReplayMemory(memory_capacity)
        self.scores = []
        self.episodes = []
        self.durations = []
        self.start = 0
        self.key = "curr"

    def init_hidden(self):
        self.hidden = self.local_Q.init_hidden()

    def select_action(self, state):
        if np.random.random() > self.eps_start:
            with torch.no_grad():
                obs = torch.tensor(state, device=device, dtype=torch.float).view(1,1,20,10)
                output, self.hidden = self.local_Q.select_action(obs, self.hidden)
                action = torch.argmax(output).item()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def learn(self):
        for _ in range(32):
            batch = list(zip(*self.replay_memory.sample()))
            
            state_batch = torch.tensor(batch[0], device=device, dtype=torch.float).unsqueeze(1)
            action_batch = torch.tensor(batch[1][:-1], device=device)
            reward_batch = torch.tensor(batch[2][:-1], device=device)
            done_batch = torch.tensor(batch[3][:-1], device=device)

            local_out, _ = self.local_Q(state_batch[:-1])
            target_out, _ = self.target_Q(state_batch[1:])
            target = local_out.clone()
            target_out = torch.max(target_out, dim=2)[0].squeeze(1)
            indexes = np.arange(len(state_batch)-1)
            old = target.clone()

            target[indexes,0,action_batch] = reward_batch + self.gamma * target_out * done_batch
            
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
            "scores": self.scores,
            "episodes": self.episodes,
            "durations": self.durations,
            "start": self.start
        }
        pickle.dump(saved, pickle_out)
        global_memory = pickle.load(open("global_memory.mm", mode="rb"))
        if not self.key in global_memory.keys():
            global_memory[self.key] = []
        global_memory[self.key] += self.replay_memory.memory
        pickle_out = open("global_memory.mm","wb")
        pickle.dump(global_memory, pickle_out)
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
        self.scores = info["scores"]
        self.episodes = info["episodes"]
        self.durations = info["durations"]
        self.start = info["start"]
        pickle_in.close()
    
    def load_memory(self, filename, n=None):
        pickle_in = open("global_memory.mm", mode="rb")
        global_memory = pickle.load(pickle_in)
        if not n:
            n = min(len(global_memory[filename]), self.replay_memory.capacity)
        for episode in global_memory[filename][-n:]:
            self.store_experience(episode)
        pickle_in.close()
        
    def store_experience(self, trajectory):
        self.replay_memory.push(trajectory)


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def contains_good_reward(self, episode):
        for _, _, reward, _ in episode:
            if reward and reward > 0:
                return True
        return False

    def push(self, episodic_trajectory):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        """
        else:
            while self.contains_good_reward(self.memory[int(self.position)]):#and np.random.random() < 0.9:
                self.position = (self.position + 1) % self.capacity
                print("skipped")
        """
        self.memory[int(self.position)] = episodic_trajectory
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.choice(self.memory)