import numpy as np
import random
import sys

SHAPES = [
    [(0,-2),(0,-1),(0,0),(0,1)],
    [(-1,-1),(0,-1),(0,0),(0,1)],
    [(0,-1),(0,0),(0,1),(-1,1)],
    [(0,-1),(-1,0),(0,0),(0,1)],
    [(0,-1),(-1,0),(0,0),(-1,1)],
    [(-1,-1),(-1,0),(0,0),(0,1)],
    [(-1,-1),(-1,0),(0,-1),(0,0)]
]

ROT_L = [
    [0,1],
    [-1,0]
]

# n E [0-3]: rotated n times
ALL_SHAPES = {
    0: [],
    1: [],
    2: [],
    3: []
}

for n in range(4):
    for shape in SHAPES:
        if shape == SHAPES[-1]:
            ALL_SHAPES[n].append(shape)
            continue
        new_shape = []
        for coor in shape:
            new_coor = coor
            for k in range(n):
                new_coor = np.dot(new_coor, ROT_L)
            new_shape.append(tuple(new_coor))
        ALL_SHAPES[n].append(new_shape)


class Environment:

    def __init__(self, row=20, col=10):
        self.row = row
        self.col = col
        self.actions = {
            0: lambda x: 0, # do nothing
            1: self._move,
            2: self._move,
            3: self._rotate,
            4: self._rotate
        }
    
    def reset(self):
        self.board = np.zeros((self.row, self.col))
        self.add_new_piece()
        self.state_history = []
        while self.is_available(self.current_piece, (1, 0)):
            self._set(num=0)
            self.rel_x += 1
            self._set(num=1)
            self.state_history.append(self.board)
        return []# self.process_history()

    def drop(self):
        while self.is_available(self.current_piece, (1, 0)):
            self._set(num=0)
            self.rel_x += 1
            self._set(num=1)

    def step(self, action):
        self.done = False
        self.reward = 0
        self.info = ""
        #self.actions[action](True if action % 2 == 0 else False)
        self.apply_gravity()
        return [], self.reward, self.done, self.info

    def add_new_piece(self, drop_point=(1,5)):
        self.rel_x, self.rel_y = drop_point
        self.rot_index = 0
        self.cur_index = np.random.randint(0,7)
        self.current_piece = ALL_SHAPES[self.rot_index][self.cur_index]
        self._set(num=1)

    def _move(self, right=False):
        to = 1 if right else -1
        if self.is_available(self.current_piece, (0,to)):
            self._set(num=0)
            self.rel_y += to
            self._set(num=1)

    def _rotate(self, reverse=False):
        to = 1 if reverse else -1
        new_rot_idx = (self.rot_index + to) % 4
        rotated_piece = ALL_SHAPES[new_rot_idx][self.cur_index]
        if self.is_available(rotated_piece, (0,0)):
            self._set(num=0)
            self.current_piece = rotated_piece
            self._set(num=1)
            self.rot_index = new_rot_idx

    def _set(self, num):
        i, j = self.rel_x, self.rel_y
        for x, y in self.current_piece:
            self.board[i+x,j+y] = num

    def is_available(self, shape, position):
        x, y = position
        k, l = self.rel_x, self.rel_y
        for i, j in shape:
            if i+x+k < 0 or i+x+k >= self.row or j+y+l < 0 or j+y+l >= self.col:
                return False
            if (i+x, j+y) not in shape and self.board[i+x+k, j+y+l] == 1:
                return False
        return True

    def apply_gravity(self):
        if self.is_available(self.current_piece, (1, 0)):
            self._set(num=0)
            self.rel_x += 1
            self._set(num=1)
        else:
            i = self.row - 1
            while i > 0:
                if np.min(self.board[i,:]) == 1:
                    row_count = 0
                    for j in range(i-1,i-5,-1):
                        if np.min(self.board[j,:]) == 1:
                            row_count += 1
                    self.reward = (row_count + 1)
                    if row_count == 3:
                        self.info = "TETRIS"
                    for j in range(i,1+row_count,-1):
                        self.board[j,:] = self.board[j-1-row_count,:]
                    i += 1
                i -= 1
            if not np.max(self.board[1]) != 0: # if done
                self.add_new_piece()
            else:
                self.reward = -1
                self.info = "done"
                self.done = True

    def process_history(self):
        if len(self.state_history) < self.frame_stack:
            processed_state = np.zeros((self.frame_stack, self.row, self.col))
            for i in range(self.frame_stack):
                processed_state[i] = self.board
            return processed_state

        processed_state = self.state_history[-self.frame_stack:]
        return processed_state