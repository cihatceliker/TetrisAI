import numpy as np
import random
import sys

SHAPES = [
    [(0,-1),(-1,0),(0,0),(0,1)],
    [(0,-1),(-1,0),(0,0),(-1,1)],
    [(-1,-1),(-1,0),(0,0),(0,1)],
    [(0,-1),(-1,1),(0,0),(0,1)],
    [(-1,-1),(0,0),(0,-1),(0,1)],
    [(-1,-1),(-1,0),(0,-1),(0,0)],
    [(0,-2),(0,0),(0,-1),(0,1)]
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
        if shape == SHAPES[-2]:
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

    def __init__(self, frame_stack, row=20, col=10):
        self.row = row
        self.col = col
        self.frame_stack = frame_stack
        self.state_history = []
        self.actions = {
            0: self.move_left,
            1: self.move_right,
            2: self.rotate_left,
            3: self.rotate_right,
            4: self.drop,
            5: lambda: 0
        }
    
    def reset(self):
        self.board = np.zeros((self.row, self.col))
        self.add_new_piece()
        return self.prepare_stack()

    def add_new_piece(self, drop_point=(1,5)):
        self.rel_x, self.rel_y = drop_point
        self.rot_index = 0
        self.cur_index = np.random.randint(0,7)
        self.current_piece = ALL_SHAPES[self.rot_index][self.cur_index]
        self._move(num=1)

    def move_right(self):
        self._move(num=0)
        if self.is_available(self.current_piece, (0,1)):
            self.rel_y += 1
        self._move(num=1)
    
    def move_left(self):
        self._move(num=0)
        if self.is_available(self.current_piece, (0,-1)):
            self.rel_y += -1
        self._move(num=1)

    def rotate_left(self):
        self._rotate(True)

    def rotate_right(self):
        self._rotate(False)

    def _rotate(self, reverse=False):
        self._move(num=0)
        to = 1 if reverse else -1
        self.rot_index = (self.rot_index + to) % 4
        rotated_piece = ALL_SHAPES[self.rot_index][self.cur_index]
        if self.is_available(rotated_piece, (0,0)):
            self.current_piece = rotated_piece
        else:
            self.rot_index -= to
        self._move(num=1)

    def _move(self, num):
        i, j = self.rel_x, self.rel_y
        for x, y in self.current_piece:
            self.board[i+x,j+y] = num

    def drop(self):
        while self.is_available(self.current_piece, (1, 0)):
            self._move(num=0)
            self.rel_x += 1
            self._move(num=1)

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
            self._move(num=0)
            self.rel_x += 1
            self._move(num=1)
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

    def step(self, action):
        self.done = False
        self.reward = 0
        self.info = ""
        self.state_history.append(self.board)
        
        self.actions[action]()
        self.apply_gravity()

        return self.prepare_stack(), self.reward, self.done, self.info

    def prepare_stack(self):
        if len(self.state_history) < self.frame_stack:
            processed_state = np.zeros((self.frame_stack, self.row, self.col))
            for i in range(self.frame_stack):
                processed_state[i] = self.board
            return processed_state
            
        processed_state = self.state_history[-self.frame_stack:]
        return processed_state