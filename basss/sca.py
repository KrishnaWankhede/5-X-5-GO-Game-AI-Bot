import numpy as np
from host import GO  # Assuming "host.py" is uploaded
import time

# Read input from a file
with open("input.txt", "r") as f:
    data = f.read().splitlines()

    # Parse the input data
    playerNumber = int(data[0])
    previousBoard = [list(map(int, data[i])) for i in range(1, 6)]
    currentBoard = [list(map(int, data[i])) for i in range(6, 11)]

# Convert board data to NumPy arrays
previousBoard = np.array(previousBoard)
currentBoard = np.array(currentBoard)

_depth = 24  # Depth of the Min-Max search
player = playerNumber  # The current player (1 or 2)

# Create a GO instance
go = GO(5)

# Initialize the board with your board data
go.init_board(5)
go.set_board(playerNumber, previousBoard, currentBoard)

# Print the currentBoard
for row in currentBoard:
    print(row)

# Define a function to find available moves
def available_moves(board):
    moves = []
    for i in range(5):
        for j in range(5):
            if board[i, j] == 0:
                moves.append((i, j))
    return moves

# Example usage:
availablemoves = available_moves(currentBoard)
#print("Available Moves:", available_moves)

# Define a function to place a piece on the board
def place_piece(board, player, position):
    i, j = position
    if board[i, j] == 0:
        board[i, j] = player
    return board
def probability_heuristic(board, player):
    # Heuristic weights
    weights = {
        "material": 1.0,
        "liberties": 0.5,
        "corners": 1.5,
        "edges": 1.0,
        "center": 0.75,
        "territory": 1.0
    }

    # Board dimensions (assuming both previousBoard and currentBoard have the same shape)
    rows, cols = board.shape

    # Initialize heuristic value
    heuristic_value = 0

    # Material Heuristic
    material_heuristic = np.sum(board == player) - np.sum(board == 3 - player)
    heuristic_value += weights["material"] * material_heuristic

    # Calculate liberties
    liberties = np.zeros_like(board)
    for i in range(rows):
        for j in range(cols):
            if board[i, j] == 0:
                liberties[i, j] = np.sum(board[i-1:i+2, j-1:j+2] == 0)

    # Liberties Heuristic
    liberties_heuristic = np.sum(liberties * (board == player))
    heuristic_value += weights["liberties"] * liberties_heuristic

    # Corners Heuristic
    corners_heuristic = np.sum(board[[0, 0, rows-1, rows-1], [0, cols-1, 0, cols-1]] == player)
    heuristic_value += weights["corners"] * corners_heuristic

    # Edges Heuristic
    edges_heuristic = np.sum(board[1:rows-1, 1:cols-1] == player)
    heuristic_value += weights["edges"] * edges_heuristic

    # Center Heuristic (assuming a 5x5 board)
    center_heuristic = np.sum(board[1:4, 1:4] == player)
    heuristic_value += weights["center"] * center_heuristic

    # Territory Heuristic (simple estimation based on stones)
    # You can implement a more sophisticated territory heuristic if needed.
    territory_heuristic = np.sum(board == player)
    heuristic_value += weights["territory"] * territory_heuristic

    return heuristic_value

# Example usage:
playerNumber = playerNumber  # Replace this with the player number (1 or 2)
best_move = None

# Iterate over the available moves and check if they are valid
for move in availablemoves:
    i, j = move  # Get the row and column from the move
    is_valid = go.valid_place_check(i, j, playerNumber, test_check=True)
    if is_valid:
        best_move = move
        break  # Stop after finding the first valid move

# if best_move is not None:
#     place_piece(currentBoard, playerNumber, best_move)
#     print("Best Move:", best_move)
# else:
#     print("No valid moves found.")

def is_terminal(board):
    # Check if the game is over by examining some conditions
    if np.sum(board == 0) == 0:
        return True
    else:
        return False
# Define a function to find the best move using Minimax with alpha-beta pruning
# def find_best_move(board, depth, player):
#     def min_max_alpha_beta(board, depth, player, alpha, beta):
#         if depth == 0 or is_terminal(board):
#             eval = probability_heuristic(board, player)
#             return eval, None  # Return the eval value and no move

#         if player == 1:  # Maximizer (player 1)
#             max_eval = -np.inf
#             best_move = None

#             for move in calculate_available_moves(board):
#                 new_board = make_move(board, move, player)
#                 eval, _ = min_max_alpha_beta(new_board, depth - 1, 2, alpha, beta)

#                 if eval is not None and eval > max_eval:
#                     max_eval = eval
#                     best_move = move

#                 alpha = max(alpha, eval)
#                 if beta <= alpha:
#                     break  # Beta cutoff

#             if best_move is None:
#                 return max_eval, None

#             return max_eval, best_move

#         else:  # Minimizer (player 2)
#             min_eval = np.inf

#             for move in calculate_available_moves(board):
#                 new_board = make_move(board, move, player)
#                 eval, _ = min_max_alpha_beta(new_board, depth - 1, 1, alpha, beta)

#                 if eval is not None and eval < min_eval:
#                     min_eval = eval

#                 beta = min(beta, eval)
#                 if beta <= alpha:
#                     break  # Alpha cutoff

#             if min_eval == np.inf:
#                 return None, None

#             return min_eval, None

#     # Call the Min-Max algorithm with alpha-beta pruning
#     best_value, best_move = min_max_alpha_beta(board, depth, player, -np.inf, np.inf)

#     if best_value is not None:
#         return best_move
#     else:
#         return None

def getMax(board, depth, player, alpha, beta):
    if depth >= _depth or is_terminal(board):
        value = probability_heuristic(board, player)
        return value, None

    available_moves = available_moves(board)
    valid_moves = []

    for move in available_moves:
        i, j = move
        is_valid = go.valid_place_check(i, j, player, test_check=True)
        if is_valid:
            valid_moves.append(move)

    best_move = None
    best_value = -np.inf

    for move in valid_moves:
        new_board = board.copy()
        new_board = place_piece(new_board, player, move)
        value, _ = getMin(new_board, depth + 1, 3 - player, alpha, beta)
        if value > best_value:
            best_value = value
            best_move = move

        alpha = max(alpha, best_value)
        if beta <= alpha:
            break  # Beta cutoff

    return best_value, best_move

def getMin(board, depth, player, alpha, beta):
    if depth >= _depth or is_terminal(board):
        value = probability_heuristic(board, player)
        return value, None

    available_moves = available_moves(board)
    valid_moves = []

    for move in available_moves:
        i, j = move
        is_valid = go.valid_place_check(i, j, player, test_check=True)
        if is_valid:
            valid_moves.append(move)

    best_move = None
    best_value = np.inf

    for move in valid_moves:
        new_board = board.copy()
        new_board = place_piece(new_board, player, move)
        value, _ = getMax(new_board, depth + 1, 3 - player, alpha, beta)
        if value < best_value:
            best_value = value
            best_move = move

        beta = min(beta, best_value)
        if beta <= alpha:
            break  # Alpha cutoff

    return best_value, best_move

# Call the function to find the best move
start = time.time()
best_move = getMax(currentBoard, 1, player, -np.inf, np.inf)
print("time taken = ", time.time() - start)
print(best_move)

# Write output to a file
with open("output.txt", "w") as f:
    if best_move[1] is None:
        f.write("PASS")
    else:
        f.write(f"{best_move[1][0]},{best_move[1][1]}")