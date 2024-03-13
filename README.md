

Overview

This script implements a Min-Max search algorithm with heuristics for playing the board game Go. It is designed to evaluate and decide the best next move for a player given the current state of the game board. The script reads the game state from an input file, calculates the best move using various strategies and heuristics, and outputs the chosen move to another file.

Input File Format

The input file (input.txt) contains the current state of the game and the player number. It is structured as follows:


The first line specifies the player number (1 for player 1 or 2 for player 2).

The next five lines represent the game board state before the opponent's last move.

The last five lines represent the game board state after the opponent's last move.

Each line representing the game board state consists of five digits (0, 1, or 2) without spaces, where:

0 indicates an empty point on the board,

1 indicates a stone placed by player 1,

2 indicates a stone placed by player 2.

Output File

The output file (output.txt) contains the coordinates of the best move determined by the script, formatted as "row,column". If no viable move is found, the script outputs "PASS".


Functionality

The script processes the game state using a combination of Min-Max search and various heuristics to evaluate potential moves. Key components include:


Game State Evaluation

Liberty and Capture Calculations: Identifies moves that increase a player's liberties (freedom to move) and captures (removal of opponent's stones).

KO Rule Enforcement: Prevents moves that would return the board to the state of two moves prior, in adherence to the KO rule in Go.

Suicide Rule Enforcement: Avoids moves that would result in the player's own stones being captured due to having no liberties.

Heuristics

Several heuristics are used to evaluate and score potential moves:



Material: Difference in the number of stones on the board.

Liberties: The freedom of a group of stones to move.

Captures: The potential to capture opponent stones.

Cluster Liberties: The collective liberties of connected groups of stones.

Territory: The area controlled by a player.

Corners and Edges: Control of corners and edges of the board, which are strategically valuable.

Center Control: Dominance in the center of the board.

Eye Formation: The formation of 'eyes' or empty points within a player's territory that cannot be filled by the opponent without being captured.

Running the Script

Ensure that the input file is formatted correctly and placed in the specified directory.

Run the script. It will read the game state from input.txt, process the information, and determine the best move.

Check output.txt for the script's chosen move.

Dependencies

This script requires Python 3 and the NumPy library for array manipulation and mathematical operations.

Conclusion

This Go game-playing script uses advanced strategies and heuristics to simulate intelligent gameplay. By evaluating the current state of the board and predicting the outcomes of various moves, it aims to select the most advantageous move for the player.

