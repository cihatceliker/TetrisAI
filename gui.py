import time
import threading
import numpy as np
import random
import pickle
from environment import Environment, ALL_SHAPES
from tkinter import Frame, Canvas, Tk
from dqn import Agent, Brain
from pyscreenshot import grab

COLORS = {
    0: "#fff",
    2: "#3e2175",
    1: "#c2abed",
    3: "#5900ff"
}

class GameGrid():

    def __init__(self, speed=0.01, size=720):
        self.draw_next_offset = size/4
        width = size / 2
        height = size + self.draw_next_offset
        self.root = Tk()
        self.root.configure(background=COLORS[0])
        self.game = Canvas(self.root, width=width, height=height, bg=COLORS[0])
        self.game.pack()
        self.env = Environment()
        self.env.reset()
        self.agent = Agent(6)
        self.agent.load("saved5500")
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
        #threading.Thread(target=self.debug_channels).start()
        #threading.Thread(target=self.watch_history).start()
        #threading.Thread(target=self.play).start()
        threading.Thread(target=self.watch_play).start()
        self.root.mainloop()

    def process_channels(self, obs):
        board_repr = np.zeros((20,10))
        board_repr[obs[2]==1] = 1
        board_repr[obs[0]==1] = 2
        board_repr[obs[1]==1] = 3
        return board_repr

    def debug_channels(self):
        for episode in reversed(self.rewarded_episode):
            for obs, _, _, _ in reversed(episode):
                self.quit = False
                while not self.quit:
                    for i in range(3):
                        self.board = obs[i]
                        self.update()
                        time.sleep(self.speed)

    def watch_play(self):
        self.action = 0
        while not self.quit:
            done = False
            state = self.env.reset()
            self.agent.init_hidden()
            while not done:
                self.action = self.agent.select_action(state)
                state, reward, done = self.env.step(self.action)
                self.board = self.process_channels(state)
                self.update()
                time.sleep(self.speed)

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
                    self.board = self.process_channels(state)
                    self.update()
                    if self.quit:
                        done = True

            trajectory.append([next_state, None, None, None])
            self.agent.store_experience(trajectory)
        pickle_out = open("aa.tt","wb")
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
                color = COLORS[curr]
                self.game.itemconfig(rect, fill=color)
        rel_x, rel_y = 2, 4
        next_piece = ALL_SHAPES[0][self.env.next_piece]
        coords = []
        for i, j in next_piece:
            coords.append((i+rel_x, j+rel_y))

        for i in range(len(self.next_piece_rectangles)):
            for j in range(len(self.next_piece_rectangles[0])):
                rect = self.next_piece_rectangles[i][j]
                if (i, j) in coords:
                    color = COLORS[3]
                else:
                    color = "#e5deff"
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
                    self.board = self.process_channels(state)
                    self.update()
                    time.sleep(self.speed)

    def init(self):
        def draw(x1, y1, sz, color, func):
            return func(x1, y1, x1+sz, y1+sz, fill=color, width=0)
        self.game_area = []
        for i in range(self.env.row):
            row = []
            for j in range(self.env.col):
                color = COLORS[0]
                rect = draw(j*self.rectangle_size, i*self.rectangle_size, 
                            self.rectangle_size, color, self.game.create_rectangle)
                row.append(rect)
            self.game_area.append(row)

        self.next_piece_rectangles = []
        rect_size = self.draw_next_offset // 4
        for i in range(8):
            row = []
            for j in range(8):
                color = COLORS[1]
                rect = draw(j*rect_size, i*rect_size+720, 
                            rect_size, color, self.game.create_rectangle)
                row.append(rect)
            self.next_piece_rectangles.append(row)

if __name__ == "__main__":
    GameGrid()