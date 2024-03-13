import numpy as np

with open("input.txt", "r") as f:
    data = f.read().splitlines()
    playerNumber = int(data[0])
    previousBoard = [list(map(int, data[i])) for i in range(1, 6)]
    currentBoard = [list(map(int, data[i])) for i in range(6, 11)]

previousBoard = np.array(previousBoard)
currentBoard = np.array(currentBoard)

_depth = 5
player = playerNumber

def available_moves(board):
    moves = []
    for i in range(5):
        for j in range(5):
            if board[i, j] == 0:
                moves.append((i, j))
    return moves

def has_liberties(board, stone):
    i, j = stone
    player = board[i, j]
    visited = np.zeros(board.shape, dtype=bool)

    def dfs(i, j):
        if i < 0 or i >= board.shape[0] or j < 0 or j >= board.shape[1] or visited[i, j]:
            return False
        if board[i, j] == 0:
            return True
        if board[i, j] != player:
            return False
        visited[i, j] = True
        return any(dfs(x, y) for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)])

    return dfs(i, j)

def remove_dead_stones(board):
    new_board = board.copy()
    for i in range(5):
        for j in range(5):
            if board[i, j] != 0 and not has_liberties(board, (i, j)):
                new_board[i, j] = 0
    return new_board

def moves_that_kill_opponent(board, player):
    moves = available_moves(board)
    killing_moves = []

    for move in moves:
        new_board = board.copy()
        new_board[move] = player
        new_board = remove_dead_stones(new_board)

        if np.count_nonzero(new_board == 3 - player) > np.count_nonzero(board == 3 - player):
            killing_moves.append(move)

    return killing_moves

def removeCapturedPieces(board, opponentNumber):
    new_board = np.array(board)
    board_size = new_board.shape[0]

    def has_liberties(stone):
        i, j = stone
        visited = np.zeros(new_board.shape, dtype=bool)

        def dfs(i, j):
            if i < 0 or i >= new_board.shape[0] or j < 0 or j >= new_board.shape[1] or visited[i, j]:
                return False
            if new_board[i, j] == 0:
                return True
            if new_board[i, j] != opponentNumber:
                return False
            visited[i, j] = True
            return any(dfs(x, y) for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)])

        return dfs(i, j)

    for i in range(board_size):
        for j in range(board_size):
            if new_board[i, j] == opponentNumber and not has_liberties((i, j)):
                new_board[i, j] = 0

    return new_board.tolist()

def removeSuicideMoves(currentBoard, playerNumber):
    newMovesAfterSuicide = []
    for i in range(5):
        for j in range(5):
            if currentBoard[i, j] == 0:
                move = (i, j)
                tempBoard = currentBoard.copy()
                tempBoard[i, j] = playerNumber
                if np.array_equal(tempBoard, removeCapturedPieces(tempBoard, playerNumber)):
                    newMovesAfterSuicide.append(move)
    return newMovesAfterSuicide

def is_ko(board, previous_board):
    return np.array_equal(board, previous_board)

def is_ko_move(move, board, previous_board):
    new_board = board.copy()
    new_board[move] = player
    return is_ko(new_board, previous_board)

def calculate_liberties(board):
    liberties = {}
    for i in range(5):
        for j in range(5):
            if board[i, j] == 0:
                liberty_count = np.sum(board[i - 1:i + 2, j - 1:j + 2] == 0)
                liberties[(i, j)] = liberty_count
    return liberties

def count_occurrences(board, player):
    return np.sum(board == player)

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

def calculate_captures(board, player):
    opponent = 3 - player
    captures = 0

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == opponent:
                if not has_liberties(board, (i, j)):
                    captures += 1

    return captures

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

    captures_heuristic = calculate_captures(board, player)
    heuristic_value += weights["captures"] * captures_heuristic

    cluster_liberties_heuristic = cluster_liberty(board, player)
    heuristic_value += weights["cluster_liberties"] * cluster_liberties_heuristic

    center_weight = 1.5
    territory_weight = 2.0
    corners_weight = 1.25
    edges_weight = 1.0


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
# Define the is_valid_move function with updates for ko rules and repeated positions
def is_valid_move(move, board, previous_board, player, liberties):
    if board[move] != 0:
        return False  # You can't place a stone where there's already one

    new_board = board.copy()
    new_board[move] = player
    new_board_after_capture = remove_dead_stones(new_board)

   
    # Check if the move is a ko move
    if is_ko_move(move, new_board_after_capture, previous_board):
        return False
    
    if not has_liberties(new_board_after_capture, move):
        return False

    # Check if the move captures opponent's stones
    opponent = 3 - player
    if np.count_nonzero(new_board_after_capture == opponent) > np.count_nonzero(board == opponent):
        return True  # The move captures opponent's stones

    # Check if the move leads to capturing your own stones
    new_board_before_capture = new_board.copy()
    new_board_before_capture = remove_dead_stones(new_board_before_capture)
    if np.count_nonzero(new_board_before_capture == player) < np.count_nonzero(new_board_after_capture == player):
        return False  # The move leads to capturing your own stones

    return True  # The move is valid
def find_best_move(currentBoard, previousBoard, player):
    depth = 5
    best_move = None
    best_value = -np.inf
    alpha = -np.inf
    beta = np.inf
    moves = available_moves(currentBoard)
    liberties = calculate_liberties(currentBoard)
    moves = removeSuicideMoves(currentBoard, player)

    for move in moves:
        if is_valid_move(move, currentBoard, previousBoard, player, liberties):
            new_board = currentBoard.copy()
            new_board[move] = player
            new_board = remove_dead_stones(new_board)
            value = min_value(new_board, previousBoard, 3 - player, depth - 1, alpha, beta)

            if is_ko(new_board, previousBoard) or is_ko_move(move, new_board, previousBoard):
                continue

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

def max_value(currentBoard, previousBoard, player, depth, alpha, beta):
    if depth == 0 or is_terminal(currentBoard):
        return probability_heuristic(currentBoard, player)

    value = -np.inf
    moves = available_moves(currentBoard)
    liberties = calculate_liberties(currentBoard)

    for move in moves:
        if is_valid_move(move, currentBoard, previousBoard, player, liberties):
            new_board = currentBoard.copy()
            new_board[move] = player
            new_board = remove_dead_stones(new_board)
            value = max(value, min_value(new_board, currentBoard, 3 - player, depth - 1, alpha, beta))

            alpha = max(alpha, value)
            if beta <= alpha:
                break

    return value

def min_value(currentBoard, previousBoard, player, depth, alpha, beta):
    if depth == 0 or is_terminal(currentBoard):
        return probability_heuristic(currentBoard, player)

    value = np.inf
    moves = available_moves(currentBoard)
    liberties = calculate_liberties(currentBoard)

    for move in moves:
        if is_valid_move(move, currentBoard, previousBoard, player, liberties):
            new_board = currentBoard.copy()
            new_board[move] = player
            new_board = remove_dead_stones(new_board)
            value = min(value, max_value(new_board, currentBoard, 3 - player, depth - 1, alpha, beta))

            beta = min(beta, value)
            if beta <= alpha:
                break

    return value

best_move = find_best_move(currentBoard, previousBoard, player)
with open("output.txt", "w") as f:
    f.write(best_move)
print(best_move)
