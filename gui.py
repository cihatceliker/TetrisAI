import time
import threading
import numpy as np
import random
import pickle
from environment import Environment
from tkinter import Frame, Canvas, Tk
from dqn import Agent, Brain
from pyscreenshot import grab

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
        self.env = Environment()
        self.env.reset()
        self.agent = Agent(6)
        self.agent.load("4300")
        #self.agent.load_memory("curr")
        self.rewarded_episode = []
        cnt = 0
        for episode in self.agent.replay_memory.memory:
            cnt += 1
            for _, _, reward, _ in episode:
                if reward and reward > 0:
                    print("selected for ", cnt)
                    self.rewarded_episode.append(episode)

        mx = 0
        for duration in self.agent.durations:
            mx = max(mx, duration)

        self.speed = speed
        self.size = size
        self.rectangle_size = size/self.env.row
        self.pause = False
        self.quit = False
        self.image_counter = 0
        self.commands = {
            113: 1, # Left
            114: 2, # Right
            53: 3, # Z
            52: 4, # X
            65: 5, # Drop
            37: 0 # Do nothing
        }
        self.init()
        self.root.title('Tetris')
        self.root.bind("<Key>", self.key_down)

        threading.Thread(target=self.watch_history).start()
        #threading.Thread(target=self.play).start()
        self.root.mainloop()

    def play(self):
        self.action = 0
        while not self.quit:
            done = False
            state = self.env.reset()
            trajectory = []
            while not done:
                if not self.pause:
                    self.pause = True
                    next_state, reward, done = self.env.step(self.action)
                    trajectory.append([state, self.action, reward, 1-done])
                    state = next_state
                    self.action = 0
                    self.board = self.env.board
                    self.update()
                    time.sleep(self.speed)
                    if self.quit:
                        done = True

            trajectory.append([next_state, None, None, None])
            self.agent.store_experience(trajectory)

        pickle_out = open("asasas"+str(np.random.random())+".tt","wb")
        pickle.dump(self.agent, pickle_out)
        pickle_out.close()

    def take_screenshot(self):
        # game windows should be on the left bottom corner
        x = 1
        y = 359
        img = grab(bbox=(x,y,x+357,y+718))
        img.save("ss"+str(self.image_counter)+".png")
        self.image_counter += 1

    def update(self):
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                rect = self.game_area[i][j]
                curr = int(self.board[i, j])
                color = BACKGROUND_COLOR if curr == 0 else PIECE_COLOR
                self.game.itemconfig(rect, fill=color)
        #self.take_screenshot()

    def key_down(self, event):
        if event.keycode == 24: # q
            self.quit = True
        if event.keycode in self.commands:
            self.action = self.commands[event.keycode]
            self.pause = False
            #action = self.commands[event.keycode]
            #self.env.actions[action][0](self.env.actions[action][1])
        #self.update()

    def watch_history(self):
        while not self.quit:
            for episode in reversed(self.rewarded_episode):
                for state, _, _, _ in episode:
                    #self.board = stacked_state[-1]
                    self.board = state
                    self.update()
                    time.sleep(self.speed)

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