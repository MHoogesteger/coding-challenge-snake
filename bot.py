from random import choice
from typing import List, Tuple

import numpy as np

from ...bot import Bot
from ...constants import Move, MOVE_VALUE_TO_DIRECTION
from ...snake import Snake

import matplotlib.pyplot as plt

from .GameState import GameState
from .navigationfunctions import a_star, straightest_to_target, determine_move_from_position, flood_count, determine_position_from_move, choose_largest_gap

DEBUG = False
np.set_printoptions(threshold=np.inf)
def crudelogger(msg):
    """
    Print a message if debug is on
    """
    if DEBUG:
        print(msg)

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
    """
    Compose a grid with the snake and other snakes
    """
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
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        super().__init__(id, grid_size)
        self.game_state = GameState(grid_size)

    @property
    def name(self):
        return 'Cherries are for Losers'

    @property
    def contributor(self):
        return 'Rinus'
    

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        g = self.game_state
        g.update_state_after_round(snake, other_snakes, candies, None)
        # plot_arena(snake, other_snakes, candies, self.grid_size)
        # compose_grid(self.grid_size, snake, other_snakes)
        if DEBUG:
            # input('ctd...')
            g.print_arena()
            g.print_snake_length_arena()
            g.print_grids()

        if len(snake.positions) > len(other_snakes[0].positions)*2:
            crudelogger("Suicide!")
            return self.suicide(snake)

        head = snake[0]
        close_candy = closest_candy(head, candies)
        nonlethal_moves = self._determine_possible_moves(snake, other_snakes[0])
        w = g.get_occupied_grid()
        path = a_star(w, head, close_candy)
        nempty = g.num_empty_spaces()
        crudelogger(f" Empty spaces: {nempty}")
        if path is None or len(path) < 2:
            crudelogger("No path found, going into panick mode")
            return choose_largest_gap(w, head, nonlethal_moves, None)
        else:
            crudelogger(path)
            move =  determine_move_from_position(head, path[1])
            if flood_count(w, determine_position_from_move(head,move), None) < nempty/3:
                move = choose_largest_gap(w, head, nonlethal_moves, None)
                crudelogger(f"Overriding A-star! ")

        
        crudelogger(f" Flood: {flood_count(w,determine_position_from_move(head,move), None)}")

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
    
    def suicide(self, snake):
        """
        Commit Suicide
        """        
        return determine_move_from_position(snake[0],snake[1])