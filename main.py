from tetris import Environment
from dqn import Agent, Brain, ReplayMemory
import numpy as np
import math
import pickle
import torch

frame_stack = 4
num_actions = 6
num_iter = 5000
env = Environment(frame_stack=frame_stack)
agent = Agent(Brain(frame_stack, num_actions), Brain(frame_stack, num_actions), num_actions)
start = 1

#agent = pickle.load(open("2280.tt", "rb"))
#start = agent.episodes[-1]+1

for episode in range(start, num_iter):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        score += reward

        agent.store_experience(state, action, reward, next_state, 1-done)
        state = next_state

        agent.learn()

    
    agent.eps_start = max(agent.eps_end, agent.eps_decay * agent.eps_start)

    agent.episodes.append(episode)
    agent.scores.append(score)

    if episode % 10 == 0:
        avg_score = np.mean(agent.scores[max(0, episode-10):(episode+1)])
        print("episode: ", episode,"score: %.6f" % score, " average score %.3f" % avg_score, "epsilon",agent.eps_start)
        if avg_score >= 300 or episode % 20 == 0:
            pickle_out = open(str(episode)+".tt","wb")
            pickle.dump(agent, pickle_out)
            pickle_out.close()
            print("weights are safe for ", episode)
    else: print("episode: ", episode,"score: %.6f" % score)