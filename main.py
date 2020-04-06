from environment import Environment
from agent import Agent, load_agent
import torch
import numpy as np
import math
import pickle
import sys

num_actions = 6
num_iter = 50000000
print_interval = 200
save_interval = 200

env = Environment()
agent = Agent(num_actions) if len(sys.argv) == 1 else load_agent(sys.argv[1])  

agent.optimizer = torch.optim.Adam(agent.local_Q.parameters(), 1e-4)
print(agent.optimizer)

for episode in range(agent.start, num_iter):
    done = False
    score = 0
    ep_duration = 0
    state, next_piece = env.reset()
    while not done:
        action = agent.select_action(state, next_piece)
        next_state, reward, done, next_next_piece, _ = env.step(action)
        agent.store_experience(state, next_piece, action, reward, next_state, next_next_piece, 1-done)
        state = next_state
        next_piece = next_next_piece
        score += reward
        ep_duration += 1
    
    agent.learn()

    agent.episodes.append(episode)
    agent.scores.append(score)
    agent.durations.append(ep_duration)
    agent.start = episode
    
    if episode % print_interval == 0 or ep_duration > 800:
        avg_score = np.mean(agent.scores[max(0, episode-print_interval):(episode+1)])
        avg_duration = np.mean(agent.durations[max(0, episode-print_interval):(episode+1)])
        if episode % save_interval == 0:
            agent.start = episode + 1
            agent.save(str(episode))
        print("Episode: %d - Avg. Duration: %d - Avg. Score: %.3f - Epsilon %.3f" % 
                    (episode, avg_duration, avg_score, agent.eps_start))
