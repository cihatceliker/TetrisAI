import os
import numpy as np
import random
import sys

EMPTY = 0.0
PIECE = 1.0

#CLEAR_REWARD = lambda x: x * 4
DEATH_REWARD = -2
DROP_CLEAR = lambda x: x * 4

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
            0: (lambda x: 0, None), # do nothing
            1: (self._move, (0,-1)),
            2: (self._move, (0,1)),
            3: (self._rotate, False),
            4: (self._rotate, True),
            5: (self._drop, None)
        }

    def reset(self):
        self.board = np.ones((self.row, self.col)) * EMPTY
        self.next_piece = np.random.randint(0,7)
        self.add_new_piece()
        self.previous_score = 0
        self.done = False
        self.previous = self.board_to_channels(self.board.copy())
        return self.process_state(), self.encode_next_piece()

    def step(self, action):
        self.reward = 0
        self.actions[action][0](self.actions[action][1])

        score = 0
        if not self._move((1,0)):
            score += self.check_complete_lines() * 2.76
            self.add_new_piece()
            
            score += self.check_rows(self.board.copy())
            self.reward += score - self.previous_score
            self.previous_score = score
        
        if action == 5:
            self.reward = DROP_CLEAR(self.reward)
        
        if self.reward < 0:
            self.reward = -((-self.reward)**0.5)
        else: self.reward = self.reward**0.5

        return self.process_state(), self.reward, self.done, self.encode_next_piece()

    def process_state(self):
        #return self.board_to_channels(self.board.copy())
        output = np.zeros((8, self.row, self.col))
        output[:4] = self.board_to_channels(self.board.copy())
        output[4:] = self.previous
        self.previous = output[:4]
        return output
    #(highest filled block) - (lowest unfilled block)
    def check_complete_lines(self):
        idxs = []
        for i in range(self.row-1, 0, -1):
            if not EMPTY in self.board[i,:]:
                idxs.append(i)
        complete_lines = len(idxs)
        for idx in reversed(idxs):
            self.board[1:idx+1,:] = self.board[0:idx,:]
        if complete_lines > 0: print("tetris", complete_lines)
        return complete_lines

    def check_rows(self, board):
        for i, j in self.current_piece:
            board[i+self.rel_x,j+self.rel_y] = EMPTY
        aggregate_height = 0
        holes = 0
        bumpiness = 0
        heights = np.zeros(self.col)
        for j in range(self.col):
            for i in range(self.row-1,0,-1):
                if board[i,j] == PIECE:
                    heights[j] = self.row - i
                    break
        aggregate_height = sum(heights)
        for j in range(self.col):
            piece_found = False
            if j < self.col-1:
                bumpiness += abs(heights[j]-heights[j+1])
            for i in range(self.row):
                if board[i,j] == PIECE:
                    piece_found = True
                if piece_found and board[i,j] == EMPTY:
                    holes += 1
        return aggregate_height * -0.51 + holes * -0.35 + bumpiness * -0.18

        """

            def check_rows(self, board):
                for i, j in self.current_piece:
                    board[i+self.rel_x,j+self.rel_y] = EMPTY
                heights = []
                for j in range(self.col):
                    for i in range(self.row-1,0,-1):
                        if board[i,j] == PIECE:
                            heights.append(self.row - i)
                            break
                aggregate_height = sum(heights)
                bumpiness = 0
                holes = 0
                for j in range(self.col):
                    piece_found = False
                    if j != self.col-2:
                        bumpiness += abs(heights[j]-heights[j+1])
                    for i in range(self.row):
                        if board[i,j] == PIECE:
                            piece_found = True
                        elif board[i,j] == EMPTY and piece_found:
                            holes += 1
                return aggregate_height * -0.51 + holes * -0.35 + bumpiness * -0.18

        """

    def board_to_channels(self, board):
        obs = np.zeros((4,self.row,self.col))
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
                obs[0, x+self.rel_x, y+self.rel_y] = 1
                obs[1, x+rel_x, y+self.rel_y] = 1
            break
        for i in range(self.row):
            for j in range(self.col):
                if board[i,j] == PIECE:
                    obs[2, i, j] = 1
                else:
                    obs[3, i, j] = 1
        return obs

    def encode_next_piece(self):
        out = np.zeros(7)
        out[self.next_piece] = 1
        return out

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
            if i+x+k >= self.row or j+y+l < 0 or j+y+l >= self.col:
                return False
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

    def _drop(self, _):
        while self._move((1,0)):
            pass