from random import choice
from typing import List, Tuple

import numpy as np

from ...bot import Bot
from ...constants import Move, MOVE_VALUE_TO_DIRECTION
from ...snake import Snake

import matplotlib.pyplot as plt

from .debugfunctions import print_character_arena_matrix, print_snake_length_arena_matrix
from .navigationfunctions import a_star, straightest_to_target, determine_move_from_position, flood_count, determine_position_from_move, choose_largest_gap

DEBUG = False

def crudelogger(msg):
    if DEBUG:
        print(msg)
class GameState:
    """
    Class that represents the game state
    """

    def __init__(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array], grid_size: Tuple[int, int]):
        self.snake = snake
        self.other_snakes = other_snakes
        self.candies = candies
        self.grid_size = grid_size

    def empty_spaces(self):
        """
        Return a list of empty spaces
        """
        return [pos for pos in np.ndindex(self.grid_size) if not collides(pos, [self.snake] + self.other_snakes)]
    
    def num_empty_spaces(self):
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

    def compose_simple_block_matrix(self) -> np.array:
        """
        Compose a matrix that represents the arena. The matrix contains the following values:
        - 0: empty
        - 1: snake
        """
        matrix = np.zeros(self.grid_size, dtype=int)
        for positions in self.snake.positions:
            matrix[positions[0], positions[1]] = 1
        for snake in self.other_snakes:
            for positions in snake.positions:
                matrix[positions[0], positions[1]] = 1
        return matrix

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


def is_on_grid(pos: np.array, grid_size: Tuple[int, int]) -> bool:
    """
    Check if a position is still on the grid
    """
    return 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]


def collides(pos: np.array, snakes: List[Snake]) -> bool:
    """
    Check if a position is occupied by any of the snakes
    """
    for snake in snakes:
        if snake.collides(pos):
            return True
    return False



def closest_candy(pos: np.array, candies: List[np.array]) -> Move:
    """
    Return the closest candy
    """
    return min(candies, key=lambda candy: np.linalg.norm(pos - candy))


def compose_grid(grid_size,snake, other_snakes,):
    grid = np.zeros((grid_size[0], grid_size[1]))
    for snake in other_snakes:
        cost = 1
        for positions in np.flip(snake.positions):
            grid[positions[0], positions[1]] = cost
            cost += 1

    plt.imshow(grid, cmap='viridis')
    plt.show()
    return 0


class CherriesAreForLosers(Bot):
    """
    Moves randomly, but makes sure it doesn't collide with other snakes
    """

    @property
    def name(self):
        return 'Cherries are for Losers'

    @property
    def contributor(self):
        return 'Rinus'
    

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        g = GameState(snake, other_snakes, candies, self.grid_size)
        # plot_arena(snake, other_snakes, candies, self.grid_size)
        # compose_grid(self.grid_size, snake, other_snakes)
        if DEBUG:
            g.print_arena()
            g.print_snake_length_arena()
        head = snake[0]
        close_candy = closest_candy(head, candies)
        nonlethal_moves = self._determine_possible_moves(snake, other_snakes[0])
        w = g.compose_simple_block_matrix()
        path = a_star(w, head, close_candy)
        nempty = g.num_empty_spaces()
        crudelogger(f" Empty spaces: {nempty}")
        if path is None or len(path) < 2:
            crudelogger("No path found, going into panick mode")
            move = choose_largest_gap(w, head, nonlethal_moves)
        else:
            crudelogger(path)
            move =  determine_move_from_position(head, path[1])
            if flood_count(w, determine_position_from_move(head,move)) < nempty/3:
                move = choose_largest_gap(w, head, nonlethal_moves)
                crudelogger(f"Overriding A-star! ")

        
        crudelogger(f" Flood: {flood_count(w,determine_position_from_move(head,move))}")

        return move

    def _determine_possible_moves(self, snake, other_snake) -> List[Move]:
        """
        Return a list with all moves that we want to do. Later we'll choose one from this list randomly. This method
        will be used during unit-testing
        """
        # highest priority, a move that is on the grid
        on_grid = [move for move in MOVE_VALUE_TO_DIRECTION
                   if is_on_grid(snake[0] + MOVE_VALUE_TO_DIRECTION[move], self.grid_size)]
        if not on_grid:
            return list(Move)

        # then avoid collisions with other snakes
        collision_free = [move for move in on_grid
                          if is_on_grid(snake[0] + MOVE_VALUE_TO_DIRECTION[move], self.grid_size)
                          and not collides(snake[0] + MOVE_VALUE_TO_DIRECTION[move], [snake, other_snake])]
        if collision_free:
            return collision_free
        else:
            return on_grid

    def choose_move(self, moves: List[Move]) -> Move:
        """
        Randomly pick a move
        """
        return choice(moves)
