import numpy as np
import sympy as sp

def parse_recursive_game(matrix, v):
    """Replace any occurrence of 'G' in the matrix with the symbolic variable v."""
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 'G':
                matrix[i][j] = v
    return matrix

def calculate_game_value(matrix):
    """Calculate the value of an m*n game given the matrix using symbolic computation."""
    m, n = matrix.shape
    
    # Create variables for the mixed strategies of Player 1 and Player 2
    p = sp.symbols(f'p0:{m}')
    q = sp.symbols(f'q0:{n}')
    
    # Convert matrix to sympy Matrix
    A = sp.Matrix(matrix)
    
    # Assume uniform strategies initially
    p = sp.Matrix([1/m] * m)
    q = sp.Matrix([1/n] * n)
    
    # Compute the expected payoff
    value_of_game = (p.T * A * q)[0]
    
    return value_of_game

def solve_recursive_game(matrix):
    """Solve the recursive game by setting up and solving the equation."""
    # Define the variable v (value of the game)
    v = sp.Symbol('v')

    # Parse the recursive game matrix, replacing 'G' with the symbolic variable v
    parsed_matrix = parse_recursive_game(matrix, v)
    
    # Calculate the value of the game (which is in terms of v)
    game_value_expression = calculate_game_value(np.array(parsed_matrix, dtype=object))
    
    # Set up the equation v = game_value_expression
    equation = sp.Eq(v, game_value_expression)
    
    # Solve the equation for v
    solution = sp.solve(equation, v)
    
    return solution

def main():
    # Example recursive game:
    # G = [[G, 2, 3], [4, G, 5], [6, 7, G]]
    matrix = [
        ['G', 2, 3],
        [4, 'G', 5],
        [6, 7, 'G']
    ]
    
    # Solve the recursive game
    solution = solve_recursive_game(matrix)
    
    print(f"Solution for the recursive game: {solution}")

if __name__ == "__main__":
    main()


