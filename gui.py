import time
import threading
import numpy as np
import random
import pickle
from environment import Environment, INFO_GROUND, INFO_NORMAL
from tkinter import Frame, Canvas, Tk
from dqn import Agent, Brain, ReplayMemory

BACKGROUND_COLOR = "#000"
PIECE_COLOR = "#fff"

class GameGrid():

    def __init__(self, speed=0.01, size=720):
        width = size / 2
        height = size
        self.root = Tk()
        self.root.configure(background=BACKGROUND_COLOR)
        self.game = Canvas(self.root, width=width, height=height, bg=BACKGROUND_COLOR)
        self.game.pack()
        self.agent = pickle.load(open("1300.tt", "rb"))
        #self.agent = Agent(Brain(4, 6), Brain(4, 6), 6)
        self.env = Environment()
        self.env.reset()

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

        #threading.Thread(target=self.watch_history).start()
        threading.Thread(target=self.watch_play).start()
        self.root.mainloop()

    def watch_play(self):

        while not self.quit:
            state = self.env.reset()
            done = False
            while not done:
                stacked_state = [state]
                info = INFO_NORMAL
                while info != INFO_GROUND:
                    action = self.agent.select_action(stacked_state)
                    #action = np.random.randint(num_actions)
                    state, reward, done, info = self.env.step(action)
                    stacked_state.append(state)
                    self.board = self.env.board
                    self.update()
                    time.sleep(self.speed)
        print("done")
        self.root.quit()

    def update(self):
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                rect = self.game_area[i][j]
                curr = int(self.board[i, j])
                color = BACKGROUND_COLOR if curr == 0 else PIECE_COLOR
                self.game.itemconfig(rect, fill=color)

    def key_down(self, event):
        if event.keycode == 24: # q
            self.quit = True
        if event.keycode in self.commands:
            #self.action = self.commands[event.keycode]
            action = self.commands[event.keycode]
            self.env.actions[action][0](self.env.actions[action][1])
        self.update()

    def watch_history(self):
        """
        for idx in self.idxs:
            self.stacked_state = self.agent.replay_memory.memory[idx][0]
            self.board = self.stacked_state[0]
            for state in self.stacked_state:
                self.board = state
                time.sleep(self.speed)
                self.update()
        """
        for stacked_state,_,_,_ in self.agent.replay_memory.memory:
            for state in stacked_state:
                #self.board = stacked_state[-1]
                self.board = state
                self.update()
                time.sleep(self.speed/1)

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


if __name__ == "__main__":
    GameGrid()