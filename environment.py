import os
import numpy as np
import random
import sys

EMPTY = 0.0
PIECE = 1.0

CLEAR_REWARD = lambda x: (x) ** 3
DEATH_REWARD = -32
DROP_CLEAR = lambda x: x * 1.2


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

    def __init__(self):
        self.row = 20
        self.col = 10
        self.actions = {
            0: (lambda x: 0, None),
            1: (self._move, (0,-1)),
            2: (self._move, (0,1)),
            3: (self._rotate, False), 
            4: (self._rotate, True),
            5: (self._drop, None)
        }

    def reset(self):
        self.reward = 0
        self.board = np.ones((self.row, self.col)) * EMPTY
        self.next_piece = np.random.randint(0,7)
        self.add_new_piece()
        self.done = False

        output = np.zeros((6, self.row, self.col))
        output[:3] = self.board_to_channels(self.board.copy())
        output[3:] = self.board_to_channels(self.board.copy())
        self.previous = output[:3]
        return output, self.encode_next_piece()

    def step(self, action):
        self.previous = self.board_to_channels(self.board.copy())

        self.reward = 0
        self.actions[action][0](self.actions[action][1])
        if not self._move((1,0)):
            self.check_rows()
            self.add_new_piece()
        
        if action == 5 and self.reward > 0:
            self.reward = DROP_CLEAR(self.reward)

        return self.process_state(), self.reward, self.done, self.encode_next_piece()

    def encode_next_piece(self):
        out = np.zeros(7)
        out[self.next_piece] = 1
        return out

    def analyze_rows(self, board):
        highest_filled = 20
        lowest_unfilled = 0
        
        cnt = 0
        found = False
        for i in range(self.row-1, 1, -1):
            for j in range(1,self.col-1):
                if board[i,j] == EMPTY \
                    and (board[i, j-1] == PIECE and board[i, j+1] == PIECE
                        or board[i-1, j] == PIECE):
                    lowest_unfilled = i
                    found = True
                    cnt += 1
            if found: break
        if not found:
            return 0

        found = False
        for i in range(self.row):
            for j in range(self.col):
                if board[i,j] == PIECE:
                    highest_filled = i
                    found = True
            if found: break
        s = (highest_filled - lowest_unfilled) * cnt
        return (-s)

    def process_state(self):
        output = np.zeros((6, self.row, self.col))
        output[:3] = self.previous
        output[3:] = self.board_to_channels(self.board.copy())
        return output
    
    def board_to_channels(self, board):
        obs = np.zeros((3,self.row,self.col))
        for i, j in self.current_piece:
            board[i+self.rel_x,j+self.rel_y] = EMPTY
        rel_x = self.rel_x
        while True:
            x, y = rel_x, self.rel_y
            if self.is_available(self.current_piece, (1,0), (rel_x, self.rel_y), board):
                rel_x += 1
                x, y = rel_x, self.rel_y
                continue
            for x, y in self.current_piece:
                obs[1, x+self.rel_x, y+self.rel_y] = 1
                obs[2, x+rel_x, y+self.rel_y] = 1
            break
        
        for i in range(self.row):
            for j in range(self.col):
                if board[i,j] == PIECE:
                    obs[0, i, j] = 1

        s = self.analyze_rows(obs[0])
        if self.reward <= 0: self.reward += -s
        return obs

    def add_new_piece(self, drop_point=(1,5)):
        self.rot_index = 0
        self.rel_x, self.rel_y = drop_point
        self.current_piece = ALL_SHAPES[self.rot_index][self.next_piece]
        self.cur_index = self.next_piece
        self.next_piece = np.random.randint(0,7)
        if not self.is_available(self.current_piece, (0, 0), (self.rel_x, self.rel_y), self.board):
            self.done = True
            self.reward = DEATH_REWARD
        else:
            self._set(num=PIECE)

    def is_available(self, shape, to, relative, board):
        x, y = to
        k, l = relative
        for i, j in shape:
            # out of bounds
            if i+x+k >= self.row or j+y+l < 0 or j+y+l >= self.col:
                return False
            # is the tile occupied by others
            if board[i+x+k, j+y+l] == PIECE:
                return False
        return True

    def _rotate(self, clockwise):
        to = 1 if clockwise else -1
        new_rot_idx = (self.rot_index + to) % 4
        rotated_piece = ALL_SHAPES[new_rot_idx][self.cur_index]
        self._set(num=EMPTY)
        if self.is_available(rotated_piece, (0,0), (self.rel_x, self.rel_y), self.board):
            self.current_piece = rotated_piece
            self.rot_index = new_rot_idx
        self._set(num=PIECE)

    def _set(self, num):
        x, y = self.rel_x, self.rel_y
        for i, j in self.current_piece:
            self.board[i+x,j+y] = num

    def _move(self, to):
        self._set(num=EMPTY)
        if self.is_available(self.current_piece, to, (self.rel_x, self.rel_y), self.board):
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
            self.reward = CLEAR_REWARD(row_count)
            print("tetris", row_count)

    def _drop(self, _):
        while self._move((1,0)):
            pass
