import heapq
import random
import numpy as np
from typing import List, Tuple
from ...snake import Snake


from ...constants import Move, MOVE_VALUE_TO_DIRECTION

def collides(pos: np.array, snakes: List[Snake]) -> bool:
    """
    Check if a position is occupied by any of the snakes
    """
    for snake in snakes:
        if snake.collides(pos):
            return True
    return False

def a_star(matrix :np.array, start :np.array, target: np.array):
    """
    A-star algorithm implementation to find the shortest path from start to target in a matrix with ones and zeros.
    """
    def getkey(coordinates):
        return coordinates[0] * matrix.shape[0] + coordinates[1]

    # Define the heuristic function as the Manhattan distance between two points
    def heuristic(a, b):
        return np.sqrt(abs(a[0] - b[0])**2 + abs(a[1] - b[1])**2)

    # Initialize the open and closed sets
    open_set = []
    closed_set = set()

    # Add the starting node to the open set
    heapq.heappush(open_set, (0, start))

    # Initialize the g and f scores
    g_score = {getkey(start): 0}
    f_score = {getkey(start): heuristic(start, target)}
    came_from = {}

    # Loop until the open set is empty
    while open_set:
        # Get the node with the lowest f score from the open set
        _, current = heapq.heappop(open_set)

        # If we've reached the target, return the path
        if (current == target).all():
            path = []
            while getkey(current) in came_from:
                path.append(current)
                current = came_from[getkey(current)]
            path.append(start)
            path.reverse()
            return path

        # Add the current node to the closed set
        closed_set.add(getkey(current))

        # Loop through the neighbors of the current node
        for neighbor in [(current[0]-1, current[1]), (current[0]+1, current[1]), (current[0], current[1]-1), (current[0], current[1]+1)]:
            # If the neighbor is not a valid point in the matrix, skip it
            if neighbor[0] < 0 or neighbor[0] >= len(matrix) or neighbor[1] < 0 or neighbor[1] >= len(matrix[0]):
                continue

            # If the neighbor is a wall, skip it
            if matrix[neighbor[0]][neighbor[1]] == 1:
                continue

            # If the neighbor is already in the closed set, skip it
            if getkey(neighbor) in closed_set:
                continue

            # Calculate the tentative g score for the neighbor
            tentative_g_score = g_score[getkey(current)] + 1

            # If the neighbor is not in the open set, add it
            if neighbor not in [x[1] for x in open_set]:
                heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor, target), neighbor))
            else:
                # If the neighbor is in the open set and the tentative g score is better than or equal to the current g score, skip it
                if tentative_g_score >= g_score[getkey(neighbor)]:
                    continue

            # Otherwise, update the g score and f score for the neighbor
            came_from[getkey(neighbor)] = current
            g_score[getkey(neighbor)] = tentative_g_score
            f_score[getkey(neighbor)] = tentative_g_score + heuristic(neighbor, target)

    # If we've exhausted all possible paths and haven't found the target, return None
    return None

def flood_count(matrix: np.array, start: Tuple[int, int], policy):
    """
    Flood fill algorithm implementation to find the largest area of zeros in a matrix.
    """
    if policy is not None:
        policy.evaluate = policy.getstate
        flood_heuristic = policy.evaluate()
        assert flood_heuristic is not None
    
    def getkey(coordinates):
        return coordinates[0] * matrix.shape[0] + coordinates[1]

    # Initialize the open and closed sets
    open_set = []
    closed_set = set()

    # Add the starting node to the open set
    heapq.heappush(open_set, (0, start))
    count = 0

    # Loop until the open set is empty
    while open_set:
        # Get the node with the lowest f score from the open set
        _, current = heapq.heappop(open_set)

        # Add the current node to the closed set
        closed_set.add(getkey(current))
        count += 1

        # Loop through the neighbors of the current node
        for neighbor in [(current[0]-1, current[1]), (current[0]+1, current[1]), (current[0], current[1]-1), (current[0], current[1]+1)]:
            # If the neighbor is not a valid point in the matrix, skip it
            if neighbor[0] < 0 or neighbor[0] >= len(matrix) or neighbor[1] < 0 or neighbor[1] >= len(matrix[0]):
                continue

            # If the neighbor is a wall, skip it
            if matrix[neighbor[0]][neighbor[1]] == 1:
                continue

            # If the neighbor is already in the closed set, skip it
            if getkey(neighbor) in closed_set:
                continue

            # If the neighbor is not in the open set, add it
            if neighbor not in [x[1] for x in open_set]:
                heapq.heappush(open_set, (0, neighbor))
    # If we've exhausted all possible paths and haven't found the target, return None
    if policy is not None:
        policy.update = policy.setstate
        policy.update(flood_heuristic)
        policy.extract = lambda x: policy.sample(sorted(x), 1)
    return count

def choose_largest_gap(matrix: np.array, head: Tuple[int, int], moves: List[Move], policy):
    """
    Choose the move that goes into the largest flood_fill area
    """
    return max(moves, key=lambda move: flood_count(matrix, determine_position_from_move(head, move), policy))

def determine_move_from_position(position, next_position):
    """
    Determine the move from the current position to the next position
    """
    if position[0] == next_position[0] - 1:
        return Move.RIGHT
    elif position[0] == next_position[0] + 1:
        return Move.LEFT
    elif position[1] == next_position[1] - 1:
        return Move.UP
    elif position[1] == next_position[1] + 1:
        return Move.DOWN
    else:
        return Move.RIGHT
    
def straightest_to_target(pos: np.array, target: np.array, moves: List[Move]) -> Move:
    """
    Return the move that brings us closest to the target
    """
    return min(moves, key=lambda move: np.linalg.norm(pos + MOVE_VALUE_TO_DIRECTION[move] - target))

def determine_position_from_move(position, move):
    """
    Determine the position from the current position and move
    """
    return position + MOVE_VALUE_TO_DIRECTION[move]


def update_strategy(policy, grid_size, candies, position, snake, other_snakes):
    found_optimal_path = False
    if policy is not None:
        count = 0
        dist = 1
        obstacles = []
        for candy in candies:
            if position[0]!=candy[0] or position[1]!=candy[1]:
                obstacles.append(candy)
        
        while not found_optimal_path:
            count += 1
            optimal_path = policy.evaluate()
            (direction,distance) = get_shortest_path_on_grid(policy,obstacles, grid_size, other_snakes[0][0])
            manh_dist = abs(position[0]-direction) + abs(position[1]-distance)
            manh_dist_other = abs(other_snakes[0][0][0]-direction) + abs(other_snakes[0][0][1]-distance)
            if manh_dist <= dist and manh_dist > 0 and not collides((direction,distance), [snake, other_snakes[0]]) and manh_dist_other > manh_dist:
                found_optimal_path = True
            if count % 1000 == 0:
                dist += 1
            if count > 10000:
                found_optimal_path = True
        policy.update(optimal_path)

def get_shortest_path_on_grid(policy,obstacles, grid_size, target):
    """Gets the shortest path on the grid to a candy"""
    # Get all free indices by distance
    free_indices = get_free_indices(obstacles, grid_size)
    # Get the shortest path to the candy
    (direction, distance) = get_shortest_path_on_grid_optimal(policy,obstacles, grid_size, free_indices, target)
    return (direction,distance)

def get_occupied_indices(obstacles, grid_size):
    return {x * grid_size[1] + y for x, y in obstacles}

def get_free_indices(obstacles, grid_size):
    return set(range(grid_size[0] * grid_size[1])) - get_occupied_indices(obstacles, grid_size)

def get_shortest_path_on_grid_optimal(policy, obstacles, grid_size, potential_paths, target):
    #TODO: Make this function more efficient and implement optimal policie for the differen paths
    # Extract a path from the policy
    index = policy.extract(potential_paths)
     # Let's find the direction of this path
    direction = index[0] // grid_size[1]
    # The distance then must be equal to the path length modulo the grid width
    distance = index[0] % grid_size[1]

    # Find if there are obstacles in the path:

    # If the direction is horizontal
    if direction == target[0]:
        # If the distance is smaller than the target distance
        if distance < target[1]:
            # If there are no obstacles in the path
            if not any(direction in obstacles for x in range(-1)):
                # Return the direction and distance
                return (direction, distance)
        # If the distance is larger than the target distance
        elif distance > target[1]:
            # If there are no obstacles in the path
            if not any(direction in obstacles for x in range(-1)):
                # Return the direction and distance
                return (direction, distance)
            
    # If the direction is vertical
    if direction == target[1]:
        # If the distance is smaller than the target distance
        if distance < target[0]:
            # If there are no obstacles in the path
            if not any(direction in obstacles for x in range(-1)):
                # Return the direction and distance
                return (direction, distance)
        # If the distance is larger than the target distance
        elif distance > target[0]:
            # If there are no obstacles in the path
            if not any(direction in obstacles for x in range(-1)):
                # Return the direction and distance
                return (direction, distance)
    
    return (direction,distance)