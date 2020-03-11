import time
import threading
import numpy as np
import random
import pickle
from tetris import Environment
from tkinter import Frame, Canvas, Tk

BACKGROUND_COLOR = "#000"
PIECE_COLOR = "#fff"

class GameGrid():

    def __init__(self, speed=0.3, size=720):
        width = size / 2
        height = size
        self.root = Tk()
        self.key = "' '"
        self.root.configure(background=BACKGROUND_COLOR)
        self.game = Canvas(self.root, width=width, height=height, bg=BACKGROUND_COLOR)
        self.game.pack()

        #self.agent = pickle.load(open("allnighter/2340.tt", "rb"))
        self.env = Environment()
        self.env.reset()
        self.speed = speed
        self.size = size
        self.rectangle_size = size/self.env.row
        self.pause = True
        self.quit = False

        self.init()
        self.root.title('Tetris')
        self.root.bind("<Key>", self.key_down)
        threading.Thread(target=self.run_game).start()
        self.root.mainloop()


    def run_game(self):
        state = self.env.reset()
        score = 0
        action = 0
        while not self.quit:
            #action = self.agent.select_action(state)
            #action = np.random.randint(5)
            next_state, reward, done, info = self.env.step(2)

            score += reward
            if done:
                self.env.reset()
            self.update()
            time.sleep(self.speed)

        print("done", score)

    def update(self):
        for i in range(self.env.row):
            for j in range(self.env.col):
                rect = self.game_area[i][j]
                curr = int(self.env.board[i, j])
                color = BACKGROUND_COLOR if curr == 0 else PIECE_COLOR
                #if i == self.env.rel_x and j == self.env.rel_y: color = "blue"
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
        key = repr(event.char)
        if key == "'q'":
            self.quit = True
        if event.keycode == 114:
            self.env._move(True)
        if event.keycode == 113:
            self.env._move(False)
        if key == "'z'":
            self.env._rotate(True)
        if key == "'x'":
            self.env._rotate(False)
        if key == "' '":
            self.env.drop()
        self.update()
        print(event)

if __name__ == "__main__":
    GameGrid()