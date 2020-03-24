import time
import threading
import numpy as np
import random
import pickle
from environment import Environment
from tkinter import Frame, Canvas, Tk
from dqn import Agent, Brain, ReplayMemory

BACKGROUND_COLOR = "#000"
PIECE_COLOR = "#fff"

class GameGrid():

    def __init__(self, speed=0.03, size=720):
        width = size / 2
        height = size
        self.root = Tk()
        self.root.configure(background=BACKGROUND_COLOR)
        self.game = Canvas(self.root, width=width, height=height, bg=BACKGROUND_COLOR)
        self.game.pack()



        self.agent = pickle.load(open("new_agent.tt", "rb"))
        #self.agent = Agent(Brain(4, 6), Brain(4, 6), 6)
        self.agent.eps_start = 0
        self.idxs = []
        for idx in range(len(self.agent.replay_memory.memory)):
            memo = self.agent.replay_memory.memory[idx]
            if memo[2] > 0:
                print(memo[2], idx)
                self.idxs.append(idx)
        
        self.env = Environment(frame_stack=4)
        self.env.reset()
        self.env.episode = 187

        self.speed = speed
        self.size = size
        self.rectangle_size = size/self.env.row
        self.pause = False
        self.quit = False
        self.commands = {
            113: 1, # Left
            114: 2, # Right
            53: 3, # Z
            52: 4, # X
            65: 5 # Drop
        }
        self.init()
        self.root.title('Tetris')
        self.root.bind("<Key>", self.key_down)
        threading.Thread(target=self.watch_play).start()
        #self.stacked_state = self.agent.replay_memory.memory[8742][3]
        #threading.Thread(target=self.watch_history).start()
        self.root.mainloop()

    def watch_history(self):
        for idx in self.idxs:
            self.stacked_state = self.agent.replay_memory.memory[idx][0]
            self.board = self.stacked_state[0]
            for state in self.stacked_state:
                self.board = state
                time.sleep(self.speed)
                self.update()

        for state,_,_,next_state,_ in self.agent.replay_memory.memory:
            #for state in self.stacked_state:
            self.board = state[-1]
            time.sleep(self.speed)
            self.update()

    def watch_play(self):
        score = 0
        for i in range(20):
            state = self.env.reset()
            #score = 0
            self.action = 0
            done = False
            while not done:
                if not self.pause:
                    self.board = state[-1]
                    self.action = self.agent.select_action(state)
                    #self.action = np.random.randint(5)
                    next_state, reward, done, info = self.env.step(self.action)
                    #print(self.action)
                    #self.action = 0
                    
                    #self.agent.store_experience(state, self.action, reward, next_state, 1-done)
                    state = next_state

                    score += 1
                    #self.pause = True
                    time.sleep(self.speed)
                    if self.quit:
                        break
                self.update()
        print("done",score)
        return
        """
            pickle_out = open("imit_agent.tt","wb")
            pickle.dump(self.agent, pickle_out)
            pickle_out.close()
            print("agent safe", score)
        """

    def update(self):
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                rect = self.game_area[i][j]
                curr = int(self.board[i, j])
                color = BACKGROUND_COLOR if curr == 0 else PIECE_COLOR
                self.game.itemconfig(rect, fill=color)

    def init(self):
        def draw(x1, y1, sz, color, func):
            return func(x1, y1, x1+sz, y1+sz, fill=color, width=0)
        # first draw the game area bg
        self.game_area = []
        for i in range(self.env.row):
            row = []
            for j in range(self.env.col):
                color = BACKGROUND_COLOR
                rect = draw(j*self.rectangle_size, i*self.rectangle_size, 
                            self.rectangle_size, color, self.game.create_rectangle)
                row.append(rect)
            self.game_area.append(row)
    
    def key_down(self, event):
        if event.keycode == 24: # q
            self.quit = True
        if event.keycode in self.commands:
            #self.action = self.commands[event.keycode]
            print("moved")
            #self.pause = False
            action = self.commands[event.keycode]
            self.env.actions[action][0](self.env.actions[action][1])
        self.update()


if __name__ == "__main__":
    GameGrid()