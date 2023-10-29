import numpy as np


from typing import List, Tuple

from ...bot import Bot
from ...constants import Move, MOVE_VALUE_TO_DIRECTION
from ...snake import Snake

import matplotlib.pyplot as plt

from .debugfunctions import print_character_arena_matrix, print_snake_length_arena_matrix

HEADGRID_ID = 0
TAILGRID_ID = 1
BODYGRID_ID = 2
LENGTHGRID_ID = 3
EASTGRID_ID = 4
NORTHGRID_ID = 5
WESTGRID_ID = 6
SOUTHGRID_ID = 7
SNAKEMULTIPLIER = SOUTHGRID_ID + 1

CANDYGRID_ID = 0
EMPTYGRID_ID = 1
SNAKE_START_ID = 2

class GameState:
    """
    Class that represents the game state
    """

    def __init__(self, grid_size: Tuple[int, int]):
        self.snake = None
        self.other_snakes = None
        self.num_snakes = None
        self.num_other_snakes = None
        self.candies = None

        self.old_snake = None
        self.old_other_snakes = []

        self.grid_size = grid_size
        self.grids = None
        self.initialized = False
        self.grid_traversal = None

    def initialize_state(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]):
        """
        Initialize the game state
        """
        self.snake = snake
        self.other_snakes = other_snakes
        self.candies = candies
        self.num_snakes = len(other_snakes) + 1
        self.num_other_snakes = len(other_snakes)
        self.initialized = True

        self.initialize_grids()

    def get_candy_grid(self):
        """
        Return the candy grid
        """
        return self.grids[:, :, CANDYGRID_ID]
    
    def get_occupied_grid(self):
        """
        Return the empty grid
        """
        return self.grids[:, :, EMPTYGRID_ID]
    
    def get_snake_grid(self):
        """
        Return the snake grid
        """
        grids = self.get_other_snake_grid(-1)
        return grids
    
    
    def get_other_snake_grid(self, index):
        """
        Return the other snake grid
        """       
        grids = self.grids[:, :, (SNAKE_START_ID + (index+1)*SNAKEMULTIPLIER):(SNAKE_START_ID +(index+2)*SNAKEMULTIPLIER)]
        return grids

    def initialize_grids(self):
        """
        Populate the snake and candy grids
        """
        self.grids = np.zeros((*self.grid_size, self.num_snakes * SNAKEMULTIPLIER + SNAKE_START_ID), dtype=int)

        initialize_snake_grids(self.get_snake_grid(), self.snake)
        for index in range(self.num_other_snakes):
            initialize_snake_grids(self.get_other_snake_grid(index), self.other_snakes[index])

        for candy in self.candies:
            self.get_candy_grid()[candy[0], candy[1]] = 1
        
        self.get_occupied_grid()[:] = np.any(self.grids[:, :, (SNAKE_START_ID + BODYGRID_ID):(SNAKE_START_ID+BODYGRID_ID+SNAKEMULTIPLIER*self.num_other_snakes+1):SNAKEMULTIPLIER], axis=2)

    def update_state_after_round(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array], order):
        """
        Update the game state after a round
        """
        if not self.initialized:
            self.initialize_state(snake, other_snakes, candies)
            return
        
        self.old_snake = self.snake
        self.old_other_snakes = self.other_snakes

        self.snake = snake
        self.other_snakes = other_snakes
        self.candies = candies

        update_snake_grids(self.get_snake_grid(), self.snake, self.old_snake)
        for index in range(self.num_other_snakes):
            update_snake_grids(self.get_other_snake_grid(index), self.other_snakes[index], self.old_other_snakes[index])

        self.get_candy_grid()[:] = 0
        for candy in self.candies:
            self.get_candy_grid()[candy[0], candy[1]] = 1

        if order is not None:
            _, self.grid_traversal = self.sort_state(self.grids,order)

        self.get_occupied_grid()[:] = np.any(self.grids[:, :, (SNAKE_START_ID + BODYGRID_ID):(SNAKE_START_ID+BODYGRID_ID+SNAKEMULTIPLIER*self.num_other_snakes+1):SNAKEMULTIPLIER], axis=2)
    
    def empty_spaces(self):
        """
        Return a list of empty spaces
        """
        return np.nonzero(self.get_occupied_grid())
    
    def num_empty_spaces(self):
        """
        Return the number of empty spaces
        """
        return len(self.empty_spaces())
        
    def compose_character_arena_matrix(self) -> np.array:
        """
        Compose a matrix that represents the arena. The matrix contains the following values:
        - 0: empty
        - 1: snake
        - 2: other snake
        - 3: candy
        """
        matrix = np.zeros(self.grid_size, dtype=int)
        for positions in self.snake.positions:
            matrix[positions[0], positions[1]] = 1
        for snake in self.other_snakes:
            for positions in snake.positions:
                matrix[positions[0], positions[1]] = 2
        for candy in self.candies:
            matrix[candy[0], candy[1]] = 3
        return matrix

    def compose_snake_length_arena_matrix(self) -> np.array:
        """
        Compose a matrix that represents the arena. The matrix contains the following values:
        - 0: empty
        - 1: snake
        - 2: other snake
        - 3: candy
        """
        matrix = np.zeros(self.grid_size, dtype=int)
        count = len(self.snake.positions)
        for positions in self.snake.positions:
            matrix[positions[0], positions[1]] = count
            count -= 1
        for snake in self.other_snakes:
            count = -len(snake.positions)
            for positions in snake.positions:
                matrix[positions[0], positions[1]] = count
                count += 1
        for candy in self.candies:
            matrix[candy[0], candy[1]] = 0
        return matrix

    def sort_state(self, state,order):
        """
        Sort the state
        """
        sorter = order._inst
        state, _, indices = tuple(reversed((sorter, sorter.uniform(state.shape[0],state.shape[1]), state)))
        return state, indices

    def print_arena(self):
        """
        Print the arena
        """
        matrix = self.compose_character_arena_matrix()
        print_character_arena_matrix(matrix)

    def print_snake_length_arena(self):
        """
        Print the arena
        """
        matrix = self.compose_snake_length_arena_matrix()
        print_snake_length_arena_matrix(matrix)

    def plot_arena(self):
        """
        Plot the arena
        """
        plt.clf()
        plt.xlim(0, self.grid_size[0])
        plt.ylim(0, self.grid_size[1])
        plt.grid()
        for positions in self.snake.positions:
            plt.scatter(positions[0], positions[1], c='r')
        for other_snake in self.other_snakes:
            for positions in other_snake.positions:
                plt.scatter(positions[0], positions[1], c='b')
        for candy in self.candies:
            plt.scatter(candy[0], candy[1], c='g')
        plt.show()

    def print_grids(self):
        """
        Print the grids
        """
        print("Candies")
        print(self.get_candy_grid())
        print("Empty")
        print(self.get_occupied_grid())
        print("Snake")
        for n in range(SNAKEMULTIPLIER):
            print(self.get_snake_grid()[:,:,n])
        for index in range(self.num_other_snakes):
            print(f"Other snake {index}")
            for n in range(SNAKEMULTIPLIER):
                print(self.get_other_snake_grid(index)[:,:,n])

def initialize_snake_grids(snake_grids, snake: Snake):
    """
    Initialize the snake grids
    """
    snake_grids[snake.positions[0][0], snake.positions[0][1],HEADGRID_ID] = 1
    snake_grids[snake.positions[-1][0], snake.positions[-1][1],TAILGRID_ID] = 1
    for index, position in enumerate(snake.positions,1):
        snake_grids[position[0], position[1],BODYGRID_ID] = 1
        snake_grids[position[0], position[1],LENGTHGRID_ID] = index
        if index < len(snake.positions):
            next_position = snake.positions[index]
            if position[0] == next_position[0]:
                if position[1] < next_position[1]:
                    snake_grids[position[0], position[1],NORTHGRID_ID] = 1
                else:
                    snake_grids[position[0], position[1],SOUTHGRID_ID] = 1
            else:
                if position[0] < next_position[0]:
                    snake_grids[position[0], position[1],EASTGRID_ID] = 1
                else:
                    snake_grids[position[0], position[1],WESTGRID_ID] = 1

def update_snake_grids(snake_grids, snake: Snake, old_snake: Snake):
    """
    Update the snake grids
    """
    head = snake.positions[0]
    tail = snake.positions[-1]
    neck = snake.positions[1]

    old_head = old_snake.positions[0]
    old_tail = old_snake.positions[-1]

    snake_grids[old_head[0], old_head[1],HEADGRID_ID] = 0
    snake_grids[head[0], head[1],HEADGRID_ID] = 1
    
    snake_grids[old_tail[0], old_tail[1],TAILGRID_ID] = 0
    snake_grids[tail[0], tail[1],TAILGRID_ID] = 1

    snake_grids[old_tail[0], old_tail[1],BODYGRID_ID] = 0
    snake_grids[tail[0], tail[1],BODYGRID_ID] = 1
    snake_grids[head[0], head[1],BODYGRID_ID] = 1

    snake_grids[:,:,LENGTHGRID_ID] = snake_grids[:,:,LENGTHGRID_ID] + (snake_grids[:,:,LENGTHGRID_ID] != 0)
    snake_grids[head[0], head[1],LENGTHGRID_ID] = 1
    snake_grids[old_tail[0], old_tail[1],LENGTHGRID_ID] = 0
    snake_grids[tail[0], tail[1],LENGTHGRID_ID] = len(snake.positions)

    snake_grids[tail[0], tail[1],EASTGRID_ID] = 0
    snake_grids[tail[0], tail[1],NORTHGRID_ID] = 0
    snake_grids[tail[0], tail[1],WESTGRID_ID] = 0
    snake_grids[tail[0], tail[1],SOUTHGRID_ID] = 0

    if head[0] == neck[0]:
        if head[1] < neck[1]:
            snake_grids[head[0], head[1],NORTHGRID_ID] = 1
        else:
            snake_grids[head[0], head[1],SOUTHGRID_ID] = 1
    else:
        if head[0] < neck[0]:
            snake_grids[head[0], head[1],EASTGRID_ID] = 1
        else:
            snake_grids[head[0], head[1],WESTGRID_ID] = 1