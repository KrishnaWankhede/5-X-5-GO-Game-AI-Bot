import numpy as np

# Read input from a file
with open("input.txt", "r") as f:
    data = f.read().splitlines()
    playerNumber = int(data[0])
    previousBoard = [list(map(int, data[i])) for i in range(1, 6)]
    currentBoard = [list(map(int, data[i])) for i in range(6, 11)]

# Convert board data to NumPy arrays
previousBoard = np.array(previousBoard)
currentBoard = np.array(currentBoard)

# Depth of the Min-Max search
_depth = 4
player = playerNumber
def getLibertyMoves(currentBoard, availableMoves, playerNumber):
    libertyMoves = []
    for move in availableMoves:
        tempBoard = np.copy(currentBoard)
        tempBoard[move] = playerNumber
        if not np.array_equal(tempBoard, removeCapturedPieces(tempBoard, 3 - playerNumber)):
            libertyMoves.append(move)
    return libertyMoves

def removeKOMoves(libertyMoves, previousBoard, availableMoves, playerNumber):
    newAvailableMoves = []
    for move in availableMoves:
        if previousBoard[move] != playerNumber:
            newAvailableMoves.append(move)
    return newAvailableMoves

def removeSuicideMoves(currentBoard, availableMoves, libertyMoves, playerNumber):
    newMovesAfterSuicide = []
    for move in availableMoves:
        if move not in libertyMoves:
            tempBoard = np.copy(currentBoard)
            tempBoard[move] = playerNumber
            if not np.array_equal(tempBoard, removeCapturedPieces(tempBoard, playerNumber)):
                newMovesAfterSuicide.append(move)
        else:
            newMovesAfterSuicide.append(move)
    return newMovesAfterSuicide

def getAllAvailableMoves(board):
    moves = []
    for i in range(5):
        for j in range(5):
            if board[i, j] == 0:
                moves.append((i, j))
    return moves

def removeLongJumpMoves(availableMoves, currentBoard, playerNumber):
    newAvailableMoves = []
    for move in availableMoves:
        if isLongJumpValid(move, currentBoard, playerNumber):
            newAvailableMoves.append(move)
    return newAvailableMoves

def isLongJumpValid(move, currentBoard, playerNumber):
    i, j = move
    if i > 1 and currentBoard[i - 1, j] == playerNumber and currentBoard[i - 2, j] == playerNumber:
        return False
    if i < 3 and currentBoard[i + 1, j] == playerNumber and currentBoard[i + 2, j] == playerNumber:
        return False
    if j > 1 and currentBoard[i, j - 1] == playerNumber and currentBoard[i, j - 2] == playerNumber:
        return False
    if j < 3 and currentBoard[i, j + 1] == playerNumber and currentBoard[i, j + 2] == playerNumber:
        return False
    return True

def removeCapturedPieces(board, playerNumber):
    newBoard = np.copy(board)
    for i in range(5):
        for j in range(5):
            if newBoard[i, j] == playerNumber:
                visited = np.zeros((5, 5), dtype=bool)
                if not hasLiberties(newBoard, i, j, visited):
                    removeStones(newBoard, i, j, playerNumber)
    return newBoard

def hasLiberties(board, i, j, visited):
    if i < 0 or i >= 5 or j < 0 or j >= 5 or visited[i, j]:
        return False
    if board[i, j] == 0:
        return True
    if board[i, j] == 3:
        return True
    visited[i, j] = True
    return (hasLiberties(board, i + 1, j, visited) or
            hasLiberties(board, i - 1, j, visited) or
            hasLiberties(board, i, j + 1, visited) or
            hasLiberties(board, i, j - 1, visited))

def removeStones(board, i, j, playerNumber):
    if i < 0 or i >= 5 or j < 0 or j >= 5:
        return
    if board[i, j] == 0:
        return
    if board[i, j] == 3:
        return
    if board[i, j] == playerNumber:
        board[i, j] = 0
        removeStones(board, i + 1, j, playerNumber)
        removeStones(board, i - 1, j, playerNumber)
        removeStones(board, i, j + 1, playerNumber)
        removeStones(board, i, j - 1, playerNumber)

def getAvailableMoves(currentBoard, previousBoard, playerNumber):
    availableMoves = getAllAvailableMoves(currentBoard)
    availableMoves = removeLongJumpMoves(availableMoves, currentBoard, playerNumber)
    libertyMoves = getLibertyMoves(currentBoard, availableMoves, playerNumber)
    availableMoves = removeKOMoves(libertyMoves, previousBoard, availableMoves, playerNumber)
    availableMoves = removeSuicideMoves(currentBoard, availableMoves, libertyMoves, playerNumber)
    return availableMoves

# Define a function to calculate liberties
def calculate_liberties(board):
    liberties = {}
    for i in range(5):
        for j in range(5):
            if board[i, j] == 0:
                liberty_count = np.sum(board[i - 1:i + 2, j - 1:j + 2] == 0)
                liberties[(i, j)] = liberty_count
    return liberties

# Define the is_valid_move function
def is_valid_move(move, board, previous_board, player, liberties):
    if move not in liberties:
        return False
    if board[move] != 0:
        return False
    new_board = board.copy()
    new_board[move] = player
    new_board_after_capture = removeCapturedPieces(new_board, 3 - player)
    if is_ko_move(move, new_board_after_capture, previous_board):
        return False
    opponent = 3 - player
    if np.count_nonzero(new_board_after_capture == opponent) > np.count_nonzero(board == opponent):
        return True
    new_board_before_capture = new_board.copy()
    new_board_before_capture = removeCapturedPieces(new_board_before_capture, player)
    if np.count_nonzero(new_board_before_capture == player) < np.count_nonzero(new_board_after_capture == player):
        return False
    return True

# Define a function to find available moves
def available_moves(board, previous_board, player):
    moves = []
    liberties = calculate_liberties(board)
    for i in range(5):
        for j in range(5):
            move = (i, j)
            if is_valid_move(move, board, previous_board, player, liberties):
                moves.append(move)
    return moves
# Define a function to check if a board state is the same as a previous board state
def is_ko(board, previous_board):
    return np.array_equal(board, previous_board)

# Define a function to check if a move is a KO move
def is_ko_move(move, board, previous_board):
    new_board = board.copy()
    new_board[move] = player

    return is_ko(new_board, previous_board)

# Define a function to calculate liberties
def calculate_liberties(board):
    liberties = {}
    for i in range(5):
        for j in range(5):
            if board[i, j] == 0:
                liberty_count = np.sum(board[i - 1:i + 2, j - 1:j + 2] == 0)
                liberties[(i, j)] = liberty_count
    return liberties

# Function to count occurrences of your pieces on the board
def count_occurrences(board, player):
    return np.sum(board == player)

# Function to find ally cluster
def cluster_liberty(board, player):
    liberties_count = 0
    checked = set()

    def dfs(r, c):
        if (r, c) in checked or board[r][c] != 0:
            return
        checked.add((r, c))
        nonlocal liberties_count
        liberties_count += 1
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                dfs(nr, nc)

    for i in range(5):
        for j in range(5):
            if board[i][j] == player:
                dfs(i, j)

    return liberties_count

# Update calculate_captures to call has_liberties correctly
def calculate_captures(board, player):
    opponent = 3 - player
    captures = 0

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == opponent:
                if not has_liberties(board, (i, j)):
                    captures += 1

    return captures

# Define the is_terminal function
def is_terminal(board):
    if np.sum(board == 0) == 0:
        return True
    else:
        return False

def center_heuristic(board, player):
    center_mask = np.array([[0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]])

    center_heuristic = np.sum(board * center_mask == player)
    return center_heuristic

def territory_heuristic(board, player):
    territory_heuristic = np.sum(board == player)
    return territory_heuristic

def corners_heuristic(board, player):
    corners_mask = np.array([[1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1]])

    corners_heuristic = np.sum(board * corners_mask == player)
    return corners_heuristic

def edges_heuristic(board, player):
    edges_mask = np.array([[0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 0, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]])

    edges_heuristic = np.sum(board * edges_mask == player)
    return edges_heuristic


# Define the is_valid_move function
# Define a function to find moves that would kill opponent's stones
def moves_that_kill_opponent(board, player):
    moves = available_moves(board, previousBoard, player)
    killing_moves = []

    for move in moves:
        new_board = board.copy()
        new_board[move] = player
        new_board = removeStones(new_board)

        if np.count_nonzero(new_board == 3 - player) > np.count_nonzero(board == 3 - player):
            killing_moves.append(move)

    return killing_moves
def probability_heuristic(board, player):
    weights = {
        "material": 1.0,
        "liberties": 1.5,
        "captures": 1.0,
        "cluster_liberties": 2.0,
        "corners": 1.25,
        "edges": 1.0,
        "center": 1.5,
        "territory": 1.0,
        "eye_formation": 1.0
    }
    
    material_heuristic = np.sum(board == player) - np.sum(board == 3 - player)
    heuristic_value = weights["material"] * material_heuristic

    liberties = np.zeros_like(board)
    for i in range(5):
        for j in range(5):
            if board[i, j] == 0:
                liberties[i, j] = np.sum(board[i - 1:i + 2, j - 1:j + 2] == 0)

    liberties_heuristic = np.sum(liberties * (board == player))
    heuristic_value += weights["liberties"] * liberties_heuristic

    # Add captures heuristic
    captures_heuristic = calculate_captures(board, player)
    heuristic_value += weights["captures"] * captures_heuristic
    
    # Add cluster liberties heuristic
    cluster_liberties_heuristic = cluster_liberty(board, player)
    heuristic_value += weights["cluster_liberties"] * cluster_liberties_heuristic

    # New heuristic functions
    center_weight = 1.5
    territory_weight = 2.0
    corners_weight = 1.25
    edges_weight = 1.0
        # Count occurrences of your pieces and assign a weight of 2.0
    player_occurrences = count_occurrences(board, player)
    heuristic_value += player_occurrences * 2.0
    center_h = center_heuristic(board, player)
    territory_h = territory_heuristic(board, player)
    corners_h = corners_heuristic(board, player)
    edges_h = edges_heuristic(board, player)

    heuristic_value += center_weight * center_h
    heuristic_value += territory_weight * territory_h
    heuristic_value += corners_weight * corners_h
    heuristic_value += edges_weight * edges_h

    eye_formation = 0
    for i in range(5):
        for j in range(5):
            if board[i][j] == player:
                neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
                empty_neighbors = sum(1 for r, c in neighbors if 0 <= r < 5 and 0 <= c < 5 and board[r][c] == 0)
                if empty_neighbors >= 2:
                    eye_formation += 1
    heuristic_value += weights["eye_formation"] * eye_formation

    return heuristic_value

# Define the find_best_move function with updates for repeated positions
def find_best_move(currentBoard, previousBoard, player):
    depth = 4
    best_move = None
    best_value = -np.inf
    alpha = -np.inf
    beta = np.inf
    moves = available_moves(currentBoard, previousBoard, player)
    liberties = calculate_liberties(currentBoard)

    for move in moves:
        if is_valid_move(move, currentBoard, previousBoard, player, liberties):
            new_board = currentBoard.copy()
            new_board[move] = player
            new_board = removeStones(new_board)
            value = min_value(new_board, previousBoard, 3 - player, depth - 1, alpha, beta)

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)
            if beta <= alpha:
                break

    if best_move is None:
        return "PASS"
    else:
        return f"{best_move[0]},{best_move[1]}"

# Define the max_value function
def max_value(currentBoard, previousBoard, player, depth, alpha, beta):
    if depth == 0 or is_terminal(currentBoard):
        return probability_heuristic(currentBoard, player)

    value = -np.inf
    moves = available_moves(currentBoard, previousBoard, player)
    liberties = calculate_liberties(currentBoard)

    for move in moves:
        if is_valid_move(move, currentBoard, previousBoard, player, liberties):
            new_board = currentBoard.copy()
            new_board[move] = player
            new_board = removeStones(new_board)
            value = max(value, min_value(new_board, currentBoard, 3 - player, depth - 1, alpha, beta))

            alpha = max(alpha, value)
            if beta <= alpha:
                break

    return value

# Define the min_value function
def min_value(currentBoard, previousBoard, player, depth, alpha, beta):
    if depth == 0 or is_terminal(currentBoard):
        return probability_heuristic(currentBoard, player)

    value = np.inf
    moves = available_moves(currentBoard, previousBoard, player)
    liberties = calculate_liberties(currentBoard)

    for move in moves:
        if is_valid_move(move, currentBoard, previousBoard, player, liberties):
            new_board = currentBoard.copy()
            new_board[move] = player
            new_board = removeStones(new_board)
            value = min(value, max_value(new_board, currentBoard, 3 - player, depth - 1, alpha, beta))

            beta = min(beta, value)
            if beta <= alpha:
                break

    return value

# Call the function to find the best move
best_move = find_best_move(currentBoard, previousBoard, player)

# Write output to a file
with open("output.txt", "w") as f:
    f.write(best_move)
print(best_move)
