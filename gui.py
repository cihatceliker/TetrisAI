import time
import threading
import numpy as np
import random
import pickle
from environment import Environment, ALL_SHAPES
from tkinter import Frame, Canvas, Tk
from agent import Agent, load_agent
import sys

COLORS = {
    0: "#fff",    # BACKGROUND
    2: "#3e2175", # SHADOW
    1: "#c2abed", # CURRENT PIECE
    3: "#5900ff"  # GROUND
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
        self.agent = load_agent(sys.argv[1])
        print(max(self.agent.durations))
        cnt = 0
        rewards = []
        for m in self.agent.replay_memory.memory:
            rewards.append(m[3])
        print(min(rewards))        
        print(max(rewards))        
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
        board_repr[obs[2]==1] = 2
        board_repr[obs[1]==1] = 1
        board_repr[obs[0]==1] = 3
        return board_repr

    def debug_channels(self):
        sample = random.sample(self.agent.replay_memory.memory, 1)
        for state, _, _, next_state, _, _ in sample:
            self.quit = False
            while not self.quit:
                for j in range(2):
                    for i in range(4):
                        #self.board = self.process_channels(state[i])
                        self.board = next_state[i+j*4]
                        self.update()
                        time.sleep(1)
            self.quit = True
            
    def watch_play(self):
        while not self.quit:
            done = False
            state, next_piece = self.env.reset()
            while not done:
                action = self.agent.select_action(state, next_piece)
                state, reward, done, next_piece = self.env.step(action)
                self.board = self.process_channels(state[:3])
                self.update()
                time.sleep(self.speed)

    def play(self):
        self.action = 0
        while not self.quit:
            done = False
            state, next_piece = self.env.reset()
            while not done:
                if not self.pause:
                    self.pause = True
                    next_state, reward, done, next_piece = self.env.step(self.action)
                    #self.agent.store_experience(state, self.action, reward, next_state, 1-done, next_piece)
                    state = next_state
                    self.action = 0
                    self.board = self.process_channels(state[:4])
                    self.update()
                    if self.quit:
                        done = True
        self.agent.save(str(np.random.random()))

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
            self.board = self.process_channels(self.play_it[4:])
            self.update()
            time.sleep(1)

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
