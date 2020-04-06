# TetrisAI

Tetris AI using DoubleDQN-CNN


The environment is a version of Tetris where 6 moves are possible:
 - Do nothing
 - Move left
 - Move right
 - Rotate
 - Rotate reverse
 - Drop the piece

# 
Input has a shape of (8,20,10). Each tile is encoded along the depth of 4 depending on whether its a part of the current piece, ground piece, shadow or if it's empty or not. Shadow is a projection of the current piece to the ground. The last 2 timesteps are stacked together to have some kind of a motion info.

# 
Here is a summary of the model
```


                                      ----- Input --
                                    /         |      \
                                   V          |       \
            Conv2d(8, 16, 5, padding=2)       |        \
                        |                     |         \
                        V                     |          \
            Conv2d(16, 24, 3, padding=1)      |           \  
                        |                     V            V
                        V
    NextPieceInfo - MaxPool2d(2) - Conv2d(8, 8, (20,1)) - Conv2d(8, 8, (1,10)) 
        
                    Concatenation
                        |
                        V
                Linear(1447, 256)
                Linear(256, 6) -> Actions

```
Gameplay after 11 hours of training on 1050ti. (Includes 4 line clear)

![alt text](/tet.gif)
# 