import numpy as np
import random
import sys

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

REWARD_FUNC = lambda x: x ** 2
DEATH_REWARD = -16

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
            0: lambda x: 0, # do nothing
            1: self._move,
            2: self._move,
            3: self._rotate,
            4: self._rotate,
            5: self._drop
        }
    
    def reset(self):
        self.board = np.zeros((self.row, self.col))
        self.add_new_piece()
        self.is_action = False
        self.done = False
        return self.board

    def step(self, action):
        self.reward = 0
        self.info = ""
        self.apply_action(action)
        #if not self.is_action:
        self.apply_gravity()
        return self.board, self.reward, self.done, self.info

    def add_new_piece(self, drop_point=(1,5)):
        self.rel_x, self.rel_y = drop_point
        self.rot_index = 0
        self.cur_index = np.random.randint(0,7)
        self.current_piece = ALL_SHAPES[self.rot_index][self.cur_index]
        
        if not self.is_available(self.current_piece, (0, 0)):
            self.done = True
            self.reward = DEATH_REWARD
            self.reset()
            print("reset")
        
        self._set(num=1)

    def is_available(self, shape, to):
        x, y = to
        k, l = self.rel_x, self.rel_y
        for i, j in shape:
            if i+x+k >= self.row or j+y+l < 0 or j+y+l >= self.col:
                return False
            print((i+x, j+y) in shape)
            if (i+x, j+y) not in shape and self.board[i+x+k, j+y+l] == 1:
                return False
        return True

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
            self.rot_index = new_rot_idx
            self._set(num=1)

    def _drop(self, _):
        while self.is_available(self.current_piece, (1, 0)):
            self._set(num=0)
            self.rel_x += 1
            self._set(num=1)

    def _set(self, num):
        x, y = self.rel_x, self.rel_y
        for i, j in self.current_piece:
            self.board[i+x,j+y] = num

    def apply_gravity(self):
        if self.is_available(self.current_piece, (1, 0)):
            self._set(num=0)
            self.rel_x += 1
            self._set(num=1)
        else:
            self.check_rows()
            self.add_new_piece()
        
    def apply_action(self, action):
        self.is_action = True
        self.actions[action](True if action % 2 == 0 else False)
        self.is_action = False

    def check_rows(self):
        row_count = 0
        i = self.row - 1
        while i > 0:
            if not 0 in self.board[i,:]:
                for j in range(i, 0, -1):
                    self.board[j,:] = self.board[j-1,:]
                i += 1
                row_count += 1
            i -= 1
        self.reward = REWARD_FUNC(row_count)
        if self.reward != 0: print(self.reward)