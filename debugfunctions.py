import numpy as np
import matplotlib.pyplot as plt

def print_character_arena_matrix(matrix: np.array):
    """
    Print the character arena matrix
    """
    grid_size = matrix.shape
    print(f' {"▁" * 3 * grid_size[0]}▁')
    for j in reversed(range(matrix.shape[1])):
        print('▕', end='')
        for i in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                print('   ', end='')
            elif matrix[i, j] == 1:
                print(' 🐍', end='')
            elif matrix[i, j] == 2:
                print(' 🪱', end='')
            elif matrix[i, j] == 3:
                print(' 🍺', end='')
            else:
                print(f' {matrix[i, j]}', end='')
        print(' ▏')
    print(f' {"▔" * 3 * grid_size[0]}▔ ')
    

    
def print_snake_length_arena_matrix(matrix: np.array):
    """
    Print the character arena matrix
    """
    grid_size = matrix.shape
    print(f' {"▁" * 3 * grid_size[0]}▁')
    for j in reversed(range(matrix.shape[1])):
        print('▕', end='')
        for i in range(matrix.shape[1]):
            s = matrix[i, j]
            if s == 0:
                print('   ', end='')
            else:
                print(f'{" " * (3-len(str(s))) + str(s)}', end='')
        print(' ▏')
    print(f' {"▔" * 3 * grid_size[0]}▔ ')