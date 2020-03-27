from environment import Environment
from dqn import Agent
import torch
import numpy as np
import math
import pickle

num_actions = 6
num_iter = 50000
print_interval = 10
save_interval = 100

env = Environment()
agent = Agent(num_actions)
agent.load("14900")
agent.optimizer = torch.optim.Adam(agent.local_Q.parameters(), 25e-5)
start = agent.start+1

for episode in range(start, num_iter):
    done = False
    score = 0
    ep_duration = 0
    state = env.reset()
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_experience(state, action, reward, next_state, 1-done)
        state = next_state
        score += reward
        ep_duration += 1

    agent.learn()

    agent.eps_start = max(agent.eps_end, agent.eps_decay * agent.eps_start)
    agent.episodes.append(episode)
    agent.scores.append(score)
    agent.durations.append(ep_duration)
    agent.start = episode

    if episode % print_interval == 0:
        avg_score = np.mean(agent.scores[max(0, episode-print_interval):(episode+1)])
        avg_duration = np.mean(agent.durations[max(0, episode-print_interval):(episode+1)])
        if episode % save_interval == 0:
            agent.save(str(episode))
        print("Episode: %d - Avg. Duration: %d - Avg. Score: %.3f - Epsilon %.3f" % 
                    (episode, avg_duration, avg_score, agent.eps_start))
