playerNumber = 2

emptyBoard = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

previousBoard = [
    [0, 1, 1, 2, 2],
    [1, 1, 1, 2, 0],
    [1, 1, 1, 2, 2],
    [1, 0, 1, 2, 0],
    [0, 1, 2, 2, 2]
]

currentBoard = [
    [0, 1, 1, 2, 2],
    [1, 1, 1, 2, 0],
    [1, 1, 1, 2, 2],
    [1, 1, 1, 2, 2],
    [0, 1, 2, 2, 2]
]

#write this to input file
with open("input.txt", "w") as f:
    f.write(str(playerNumber) + "\n")
    for row in previousBoard:
        f.write("".join(map(str, row)) + "\n")
    for row in currentBoard:
        f.write("".join(map(str, row)) + "\n")