import numpy as np
from scipy.linalg import inv
from scipy.optimize import linprog

def check_saddle_point(matrix):
    row_mins = np.min(matrix, axis=1)
    max_of_row_mins = np.max(row_mins)

    col_maxs = np.max(matrix, axis=0)
    min_of_col_maxs = np.min(col_maxs)

    if max_of_row_mins == min_of_col_maxs:
        return True, max_of_row_mins
    return False, None

def remove_dominated_strategies(matrix):
    rows_to_keep = list(range(matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            if all(matrix[i] >= matrix[j]) and any(matrix[i] > matrix[j]):
                rows_to_keep.remove(j)
            elif all(matrix[j] >= matrix[i]) and any(matrix[j] > matrix[i]):
                rows_to_keep.remove(i)
    matrix = matrix[rows_to_keep, :]

    cols_to_keep = list(range(matrix.shape[1]))
    for i in range(matrix.shape[1]):
        for j in range(i + 1, matrix.shape[1]):
            if all(matrix[:, i] <= matrix[:, j]) and any(matrix[:, i] < matrix[:, j]):
                cols_to_keep.remove(j)
            elif all(matrix[:, j] <= matrix[:, i]) and any(matrix[:, j] < matrix[:, i]):
                cols_to_keep.remove(i)
    matrix = matrix[:, cols_to_keep]

    return matrix

def solve_2x2_matrix(matrix):
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]

    p1 = (d - c) / ((d - c) + (a - b))
    p2 = 1 - p1
    q1 = (d - b) / ((d - c) + (a - b))
    q2 = 1 - q1
    value_of_game = (a * d - b * c) / ((a + d) - (b + c))

    return value_of_game, [p1, p2], [q1, q2]

def solve_nxn_matrix(matrix):
    A_inv = inv(matrix)
    ones = np.ones((matrix.shape[0], 1))
    denom = np.dot(np.dot(ones.T, A_inv), ones)[0][0]

    p = np.dot(ones.T, A_inv).flatten() / denom
    q = np.dot(A_inv, ones).flatten() / denom
    value_of_game = 1 / denom

    return value_of_game, p, q

def solve_large_matrix(matrix):
    m, n = matrix.shape

    # For Player 1 (row player)
    c = [-1] + [0] * m
    A_ub = np.hstack([np.ones((n, 1)), -matrix.T])
    b_ub = np.zeros(n)
    A_eq = np.array([[0] + [1] * m])
    b_eq = np.array([1])
    bounds = [(None, None)] + [(0, 1)] * m
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        raise ValueError("Linear programming failed to find a solution for Player 1.")

    v = 1 / res.fun  # Value of the game
    p1_strategy = res.x[1:] * v

    # For Player 2 (column player)
    c = [1] + [0] * n
    A_ub = np.hstack([-np.ones((m, 1)), matrix])
    b_ub = np.zeros(m)
    A_eq = np.array([[0] + [1] * n])
    b_eq = np.array([1])
    bounds = [(None, None)] + [(0, 1)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        raise ValueError("Linear programming failed to find a solution for Player 2.")

    q2_strategy = res.x[1:] * v

    return v, p1_strategy, q2_strategy

def main():
    m = int(input("Enter the number of strategies for Player 1: "))
    n = int(input("Enter the number of strategies for Player 2: "))

    print(f"Enter the {m}x{n} payoff matrix (Player 1's payoffs):")
    matrix = np.zeros((m, n))
    for i in range(m):
        row = input(f"Enter row {i + 1} (space-separated): ").split()
        matrix[i] = list(map(float, row))

    # Remove dominated strategies
    reduced_matrix = remove_dominated_strategies(matrix)
    print(f"Reduced payoff matrix after removing dominated strategies:\n{reduced_matrix}")

    saddle, value = check_saddle_point(reduced_matrix)
    if saddle:
        print(f"Saddle point found! Value of the game: {value}")
        print("Pure strategies are optimal.")
    else:
        try:
            if reduced_matrix.shape == (2, 2):  # 2x2 matrix
                value, prob_p1, prob_p2 = solve_2x2_matrix(reduced_matrix)
            elif reduced_matrix.shape[0] == reduced_matrix.shape[1]:  # n x n matrix
                value, prob_p1, prob_p2 = solve_nxn_matrix(reduced_matrix)
            else:  # Non-square matrix
                value, prob_p1, prob_p2 = solve_large_matrix(reduced_matrix)

            print("No saddle point found. Solving for mixed strategy equilibrium.")
            print(f"Value of the game: {value}")
            print(f"Optimal mixed strategy for Player 1: {prob_p1}")
            print(f"Optimal mixed strategy for Player 2: {prob_p2}")
        except ValueError as e:
            print(e)

if __name__ == "__main__":
    main()

