import time
import threading
import numpy as np
import random
import pickle
from environment import Environment
from tkinter import Frame, Canvas, Tk
#from dqn import Agent, Brain

BACKGROUND_COLOR = "#000"
PIECE_COLOR = "#fff"

class GameGrid():

    def __init__(self, speed=0.3, size=720):
        width = size / 2
        height = size
        self.root = Tk()
        self.root.configure(background=BACKGROUND_COLOR)
        self.game = Canvas(self.root, width=width, height=height, bg=BACKGROUND_COLOR)
        self.game.pack()


        #self.agent = Agent(Brain(4, 6), Brain(4, 6), 6)
        #self.load_agent("trained.tt")
        #self.agent.eps_start = 0

        self.env = Environment(frame_stack=4)
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
        threading.Thread(target=self.run_game).start()
        self.root.mainloop()
        
    def load_agent(self, file): self.agent = pickle.load(open(file, "rb"))

    def run_game(self):
        while not self.quit:
            state = self.env.reset()
            score = 0
            self.action = 0
            done = False
            while not done:
                if not self.pause:
                    #self.action = self.agent.select_action(state)
                    #action = np.random.randint(5)
                    next_state, reward, done, info = self.env.step(self.action)
                    #print(self.action)
                    #self.action = 0
                    
                    #self.agent.store_experience(state, self.action, reward, next_state, 1-done)
                    state = next_state

                    score += reward
                    #self.pause = True
                    time.sleep(self.speed)
                    if self.quit:
                        break
                self.update()
        return
        """
            pickle_out = open("imit_agent.tt","wb")
            pickle.dump(self.agent, pickle_out)
            pickle_out.close()
            print("agent safe", score)
        """

    def update(self):
        for i in range(self.env.row):
            for j in range(self.env.col):
                rect = self.game_area[i][j]
                curr = int(self.env.board[i, j])
                color = BACKGROUND_COLOR if curr == 0 else PIECE_COLOR
                if i == self.env.rel_x and j == self.env.rel_y: color = "#999"
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