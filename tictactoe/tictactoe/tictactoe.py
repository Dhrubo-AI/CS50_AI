"""
Tic Tac Toe Player
"""

import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x = 0
    o = 0
    for i in board:
        for j in i:
            if j==X:
                x+=1
            elif j==O:
                o+=1
    if x==o:
        return X
    else:
        return O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    row = 0
    for i in board:
        col = 0
        for j in i:
            if j == EMPTY:
                actions.add((row,col))
            col+=1
        row+=1
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    bard = copy.deepcopy(board)
    play = player(bard)
    (i,j) = action
    bard[i][j] = play
    return bard

def check(list):
    if list.count(X)==3:
        return X
    elif list.count(O)==3:
        return O
    else:
        return None

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    num = 0
    leftdiag = []
    rightdiag = []
    verts = [[],[],[]]
    for row in board:
        out = check(row)
        if out!=None:
            return out
        leftdiag.append(board[num][num])
        rightdiag.append(board[num][2-num])
        for i in range(3):
            verts[i].append(row[i])
        num+=1
    verts.append(leftdiag)
    verts.append(rightdiag)
    for col in verts:
        out = check(col)
        if out!=None:
            return out
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True
    for i in board:
        for j in i:
            if j==EMPTY:
                return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win==X:
        return 1
    if win==O:
        return -1
    return 0

def optimal(board):
    if terminal(board):
        return utility(board)
    turn = player(board)
    action = actions(board)
    if turn=="X":
        a = []
        for act in action:
            val = optimal(result(board,act))
            if val==1:
                return val
            a.append(val)
        return max(a)
    if turn=="O":
        b = []
        for act in action:
            val = optimal(result(board,act))
            if val==-1:
                return val
            b.append(val)
        return min(b)

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    play = player(board)
    action = actions(board)
    op = ()
    l ={}
    for act in action:
        val=optimal(result(board,act))
        l[act]=val
        if val==1 and play=="X":
            return act
        if val==-1 and play=="O":
            return act
        if val==0:
            op = act
    return op


