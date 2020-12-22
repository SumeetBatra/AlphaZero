import numpy as np
import chess
import torch

uci_to_numerical = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}

dir_to_numerical = {'N': 1, 'NE': 2, 'E': 3, 'SE': 4, 'S': 5, 'SW': 6, 'W': 7, 'NW': 8}

# 8 possible knight moves (dx, dy) to their corresponding plane indices
kdelta_to_planeidx = {(2, 1): 56, (1, 2): 57, (-1, 2): 58, (-2, 1): 59,
                      (-2, -1): 60, (-1, -2): 61, (1, -2): 62, (2, -1): 63}

# 3 possible pawn moves (dx, dy) to a numerical value
pdelta_to_numerical = {(1, 1): 1, (0, 1): 2, (-1, 1): 3,
                       (-1, -1): 1, (0, -1): 2, (1, -1): 3}  # black pawn underpromotion should mirror white's

# 3 possible underpromotions (knight, bishop, rook) to a numerical value
underpromotion_to_numerical = {'n': 1, 'b': 2, 'r': 3}

# dicts that go in the opposite direction
numerical_to_uci = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h'}

numerical_to_dir = {1: 'N', 2: 'NE', 3: 'E', 4: 'SE', 5: 'S', 6: 'SW', 7: 'W', 8: 'NW'}

planeidx_to_kdelta = {56: (2, 1), 57: (1, 2), 58: (-1, 2), 59: (-2, 1),
                      60: (-2, -1), 61: (-1, -2), 62: (1, -2), 63: (2, -1)}

numerical_to_underpromotion = {1: 'n', 2: 'b', 3: 'r'}

numerical_to_pdelta = {1: (1, 1), 2: (0, 1), 3: (-1, 1)}


def encode_actions(str_board, legal_move, flattened=False):
    '''
    encode a legal move into a representation that matches the probs output of the policy head
    :param str_board: (8,8) string representation of chess.Board
    :param legal_move: string representation of a legal move
    :return: (i,j) pos of the piece to pick up and plane_idx (k)
    '''
    init_pos = np.array([uci_to_numerical[legal_move[0]]-1, int(legal_move[1])-1])
    final_pos = np.array([uci_to_numerical[legal_move[2]]-1, int(legal_move[3])-1])
    dx, dy = final_pos - init_pos
    num_squares = abs(dx) or abs(dy)

    pawn_underpromote = False
    if len(legal_move) == 5 and legal_move[-1] not in ['q', 'Q']:
        pawn_underpromote = True
    if pawn_underpromote:
        promote_idx = ((underpromotion_to_numerical[legal_move[-1]] - 1) * 3 + pdelta_to_numerical[(dx, dy)]) - 1
        stack_idx = 64 + promote_idx
        return init_pos[0], init_pos[1], stack_idx

    if str_board[-init_pos[1]-1, init_pos[0]] not in ['n', 'N']:  # piece to move is not a knight. Also (row, col) = (dy, dx)
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
        stack_idx = ((num_squares - 1) * 8 + dir) - 1
    else:
        # piece to move is a knight
        stack_idx = kdelta_to_planeidx[(dx, dy)]

    if flattened:
        return init_pos[0] * 8 + init_pos[1] * 8 + stack_idx * 73
    return init_pos[0], init_pos[1], stack_idx


def decode_action(action, board):
    # TODO: Test to make sure this works
    action_space = np.zeros(4672)
    action_space[action] = 1
    action_space = action_space.reshape((8, 8, 73))
    i, j, k = np.argwhere(action_space).squeeze()
    numerical_board = np.array(str(board).split()).reshape(8,  8)
    piece_type = numerical_board[-j-1, i]  # (row, col) = (dy, dx)
    start_row = numerical_to_uci[i+1]
    start_col = j+1
    start_pos = start_row + str(start_col)
    if k <= 55:  # Queen move in compass dir
        dir = numerical_to_dir[(k % 8) + 1]
        num_squares = (k // 8) + 1
        if dir == 'N':
            dx, dy = 0, num_squares
        elif dir == 'S':
            dx, dy = 0, -num_squares
        elif dir == 'E':
            dx, dy = num_squares, 0
        elif dir == 'W':
            dx, dy = -num_squares, 0
        elif dir == 'NE':
            dx, dy = num_squares, num_squares
        elif dir == 'NW':
            dx, dy = -num_squares, num_squares
        elif dir == 'SE':
            dx, dy = num_squares, -num_squares
        else:
            dx, dy = -num_squares, -num_squares

        end_row = chr(ord(start_row) + dx)
        end_col = str(start_col + dy)
        end_pos = end_row + end_col
        if piece_type in ['p', 'P'] and int(end_pos[-1]) in [1, 8]:  # pawn promotion to queen
            end_pos += 'q'
        move = chess.Move.from_uci(start_pos + end_pos)
    elif 56 <= k <= 63:  # knight move
        dx, dy = planeidx_to_kdelta[k]
        end_row = chr(ord(start_row) + dx)
        end_col = str(start_col + dy)
        end_pos = end_row + end_col
        move = chess.Move.from_uci(start_pos + end_pos)
    else:  # pawn underpromotion
        underpromote = numerical_to_underpromotion[((k - 64) // 3) + 1]
        dx, dy = numerical_to_pdelta[((k - 64) % 3) + 1]
        if board.turn == chess.BLACK:
            dx, dy = -dx, -dy
        end_row = chr(ord(start_row) + dx)
        end_col = str(start_col + dy)
        end_pos = end_row + end_col
        move = chess.Move.from_uci(start_pos + end_pos + underpromote)

    return move


def mask_illegal_actions(state, p_raw):
    # TODO: Test to make sure this works correctly
    legal_moves = [mv.uci() for mv in state.legal_moves]
    mask = np.zeros(4672).reshape((8, 8, 73))
    p_raw = p_raw.reshape((8, 8, 73))
    for legal_move in legal_moves:
        str_board = np.array(str(state).split()).reshape(8, 8)
        i, j, k = encode_actions(str_board, legal_move)
        # test = np.zeros((8, 8, 73))
        # test[i, j, k] = 1
        # test = test.reshape(-1)
        # test_act = decode_action(np.argwhere(test), state)
        # assert(str(test_act) == legal_move)
        mask[i, j, k] = 1

    p_masked = p_raw * torch.Tensor(mask).detach()
    return p_masked
