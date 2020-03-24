from environment import Environment, INFO_NORMAL, INFO_GROUND
from dqn import Agent, ReplayMemory, Brain
import torch
import numpy as np
import math
import pickle

num_actions = 5
num_iter = 5000

env = Environment()
agent = Agent(num_actions)
#agent = pickle.load(open("300.tt", mode="rb"))
start = agent.start

#for m in agent.replay_memory.memory: print(len(m[1]), len(m[0]))

total_duration = 0
for episode in range(start, num_iter):
    done = False
    score = 0
    ep_duration = 0
    state = env.reset()

    while not done:
        stacked_state = [state]
        actions = []
        info = INFO_NORMAL

        while info != INFO_GROUND:
            action = agent.select_action(stacked_state)
            #action = np.random.randint(num_actions)
            state, reward, done, info = env.step(action)

            actions.append(action)
            stacked_state.append(state)

            ep_duration += 1

        agent.store_experience(stacked_state, actions, reward, 1-done)
        agent.learn()
        score += reward

    agent.eps_start = max(agent.eps_end, agent.eps_decay * agent.eps_start)
    agent.episodes.append(episode)
    agent.scores.append(score)
    agent.durations.append(ep_duration)
    agent.start = episode

    total_duration += ep_duration

    if episode % 10 == 0:

        avg_score = np.mean(agent.scores[max(0, episode-10):(episode+1)])
        print("episode: ", episode, "duration:", total_duration,"score: %.6f" % score, \
            " average score %.3f" % avg_score, "epsilon %.4f"%agent.eps_start)
        total_duration = 0
    
        if episode % 100 == 0:
            pickle_out = open(str(episode)+".tt","wb")
            pickle.dump(agent, pickle_out)
            pickle_out.close()
    #else: print("episode: ", episode,"duration", ep_duration,"score: %.6f" % score)
