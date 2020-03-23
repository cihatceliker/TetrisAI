from environment import Environment
from dqn import Agent, ReplayMemory
import numpy as np
import math
import pickle
import torch

num_actions = 5
num_iter = 5000
frame_stack = 4

env = Environment(frame_stack=frame_stack)

agent = Agent(frame_stack, num_actions)
start = 1

#agent = pickle.load(open("".tt", mode="rb"))
#start = agent.episodes[-1]+1
#env.episode = agent.env_episodes[-1]

total_duration = 0
for episode in range(start, num_iter):
    done = False
    score = 0
    ep_duration = 0
    state = env.reset()
    agent.env_episodes.append(env.episode)

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_experience(state, action, reward, next_state, 1-done)
        state = next_state
        agent.learn()
        score += reward
        ep_duration += 1

    agent.eps_start = max(agent.eps_end, agent.eps_decay * agent.eps_start)
    
    agent.episodes.append(episode)
    agent.scores.append(score)
    agent.durations.append(ep_duration)

    total_duration += ep_duration

    if episode % 10 == 0:
        cg = 0
        if episode % 1000000 == 0:
            for memo in agent.replay_memory.memory:
                if memo[2] > 0:
                    cg += 1
        avg_score = np.mean(agent.scores[max(0, episode-10):(episode+1)])
        print("duration:", total_duration, "episode: ", episode,"env episode: ", env.episode,"score: %.6f" % score, \
            " average score %.3f" % avg_score, "epsilon %.4f"%agent.eps_start, "goods", cg)
        total_duration = 0
        if episode % 50 == 0:
            pickle_out = open(str(episode)+".tt","wb")
            pickle.dump(agent, pickle_out)
            pickle_out.close()
    #else: print("episode: ", episode,"score: %.6f" % score)