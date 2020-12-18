import numpy as np

uci_to_numerical = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}

dir_to_numerical = {'N': 1, 'NE': 2, 'E': 3, 'SE': 4, 'S': 5, 'SW': 6, 'W': 7, 'NW': 8}

# 8 possible knight moves (dx, dy) to their corresponding plane indices
kdelta_to_planeidx = {(2, 1): 56, (1, 2): 57, (-1, 2): 58, (-2, 1): 59,
                      (-2, -1): 60, (-1, -2): 61, (1, -2): 62, (2, -1): 63}

# 3 possible pawn moves (dx, dy) to a numerical value
pdelta_to_numerical = {(1, 1): 1, (0, 1): 2, (-1, 1): 3}

# 3 possible underpromotions (knight, bishop, rook) to a numerical value
underpromotion_to_numerical = {'k': 1, 'K': 1, 'b': 2, 'B': 2, 'r': 3, 'R': 3}



def encode_action(str_board, legal_move):
    '''
    encode a legal move into a representation that matches the probs output of the policy head
    :param str_board: (8,8) string representation of chess.Board
    :param legal_move: string representation of a legal move
    :return: (i,j) pos of the piece to pick up and plane_idx (k)
    '''
    init_pos = np.array([uci_to_numerical[legal_move[0]], int(legal_move[1])])
    final_pos = np.array([uci_to_numerical[legal_move[2]], int(legal_move[3])])
    dx, dy = final_pos - init_pos
    num_squares = dx or dy

    pawn_underpromote = False
    if len(legal_move) == 5 and legal_move[-1] not in ['q', 'Q']:
        pawn_underpromote = True
    if pawn_underpromote:
        promote_idx = underpromotion_to_numerical[legal_move[-1]] * pdelta_to_numerical[(dx, dy)] - 1
        stack_idx = 64 + promote_idx
        return init_pos[0], init_pos[1], stack_idx

    if str_board[init_pos] not in ['n', 'N']:  # piece to move is not a knight
        direction = None
        if dx and not dy:
            if dx > 0:
                direction = 'E'
            else:
                direction = 'W'
        elif dy and not dx:
            if dy > 0:
                direction = 'N'
            else:
                direction = 'S'
        elif dx and dy:
            if dx > 0 and dy > 0:
                direction = 'NE'
            elif dy < 0 < dx:
                direction = 'SE'
            elif dx < 0 < dy:
                direction = 'NW'
            else:
                direction = 'SW'
        dir = dir_to_numerical[direction]
        stack_idx = (dir * num_squares) - 1
    else:
        # piece to move is a knight
        stack_idx = kdelta_to_planeidx[(dx, dy)]

    return init_pos[0], init_pos[1], stack_idx


def mask_illegal_actions(state, p_raw):
    # TODO: Test to make sure this works correctly
    legal_moves = [mv.uci() for mv in state.legal_moves]
    mask = np.zeros(4672).reshape((8, 8, 73))
    p_raw = p_raw.reshape((8, 8, 73))
    for legal_move in legal_moves:
        str_board = np.array(str(state).split()).reshape(8, 8)
        i, j, k = encode_action(str_board, legal_move.uci())
        mask[i, j, k] = 1

    p_masked = p_raw[mask == 0] = 0
    return p_masked
