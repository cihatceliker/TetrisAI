import os
import numpy as np
import random
import sys

EMPTY = 0.0
PIECE = 1.0

REWARD_FUNC = lambda x: x
DEATH_REWARD = -1
DEFAULT_REWARD = 0
NON_DROP = -1e-4
DROP_CLEAR = 0.2


# ARS rotation
SHAPES = {
    0: [
        [(-1,-1),(-1,0),(0,-1),(0,0)]   # O
    ],
    2: [
        [(0,-2),(0,-1),(0,0),(0,1)],    # I
        [(-1,-1),(-1,0),(0,0),(0,1)],   # Z
        [(0,-1),(-1,0),(0,0),(-1,1)]    # Z'
    ],
    4: [
        [(-1,-1),(0,-1),(0,0),(0,1)],   # L'
        [(0,-1),(0,0),(0,1),(-1,1)],    # L
        [(0,-1),(-1,0),(0,0),(0,1)]     # T
    ]
}
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

def rotate_n_times(shape, n):
    new_shape = []
    for coor in shape:
        new_coor = coor
        for k in range(n):
            new_coor = np.dot(new_coor, ROT_L)
        new_shape.append(tuple(new_coor))
    return new_shape

for n in range(4):
    ALL_SHAPES[n].append(SHAPES[0][0])
    for shape in SHAPES[2]:
        ALL_SHAPES[n].append(rotate_n_times(shape, n%2))
    for shape in SHAPES[4]:
        ALL_SHAPES[n].append(rotate_n_times(shape, n))


class Environment:

    def __init__(self, row=20, col=10):
        self.row = row
        self.col = col
        self.actions = {
            0: (lambda x: 0, None),
            1: (self._move, (0,-1)),
            2: (self._move, (0,1)),
            3: (self._rotate, False), 
            4: (self._rotate, True),
            5: (self._drop, None)
        }

    def reset(self):
        self.board = np.ones((self.row, self.col)) * EMPTY
        self.add_new_piece()
        self.done = False
        return self.process_state()

    def step(self, action):
        self.reward = 0
        self.actions[action][0](self.actions[action][1])
        if not self._move((1,0)):
            self.check_rows()
            self.add_new_piece()

        if action == 5:
            if self.reward > 0:
                print("dropped and cleared")
                self.reward += DROP_CLEAR
        else:
            # tiny negative reward to make it drop asap
            self.reward += NON_DROP

        return self.process_state(), self.reward, self.done

    def process_state(self):
        cp = self.board.copy()
        for x, y in self.current_piece:
            cp[x+self.rel_x, y+self.rel_y] = -1
        return cp

    def add_new_piece(self, drop_point=(1,5)):
        self.rel_x, self.rel_y = drop_point
        self.rot_index = 0
        self.cur_index = np.random.randint(0,7)
        self.current_piece = ALL_SHAPES[self.rot_index][self.cur_index]
        if not self.is_available(self.current_piece, (0, 0)):
            self.done = True
            self.reward = DEATH_REWARD
        else:
            self._set(num=PIECE)

    def is_available(self, shape, to):
        x, y = to
        k, l = self.rel_x, self.rel_y
        for i, j in shape:
            # out of bounds
            if i+x+k >= self.row or j+y+l < 0 or j+y+l >= self.col:
                return False
            # is the tile occupied by others
            if self.board[i+x+k, j+y+l] == PIECE:
                return False
        return True

    def _rotate(self, clockwise):
        to = 1 if clockwise else -1
        new_rot_idx = (self.rot_index + to) % 4
        rotated_piece = ALL_SHAPES[new_rot_idx][self.cur_index]
        self._set(num=EMPTY)
        if self.is_available(rotated_piece, (0,0)):
            self.current_piece = rotated_piece
            self.rot_index = new_rot_idx
        self._set(num=PIECE)

    def _set(self, num):
        x, y = self.rel_x, self.rel_y
        for i, j in self.current_piece:
            self.board[i+x,j+y] = num

    def _move(self, to):
        self._set(num=EMPTY)
        if self.is_available(self.current_piece, to):
            self.rel_x += to[0]
            self.rel_y += to[1]
            self._set(num=PIECE)
            return True
        self._set(num=PIECE)
        return False

    def check_rows(self):
        row_count = 0
        idxs = []
        for i in range(self.row-1, 0, -1):
            if not EMPTY in self.board[i,:]:
                idxs.append(i)
        row_count = len(idxs)
        for idx in reversed(idxs):
            self.board[1:idx+1,:] = self.board[0:idx,:]
        if row_count != 0:
            self.reward = REWARD_FUNC(row_count)
            print("tetris", row_count)
            #os.system("audacious " + "clicksound.wav")

    def _drop(self, _):
        while self._move((1,0)):
            pass