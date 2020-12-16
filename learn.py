import torch
import chess
import numpy as np
import torch.nn as nn

import chess_env


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOTAL_STEPS = 7e5
TIMESTEPS = 8


def board_to_layers(boards, color, total_moves, w_castling, b_castling, no_progress_count):
    '''
    convert chess.Board to binary layers to feed into a neural network.
    :param env: ChessEnv
    :param boards: chess.Board game state
    :param color: white's turn (True) or black's turn (False)
    :param total_moves: total moves elapsed in this game
    :param w_castling: legality of white castling kingside or queenside
    :param b_castling: legality of black castling kingside or queenside
    :param no_progress_count: number of moves without progress (50 moves without progress is automatic draw)
    :return: torch Tensor of all layers stacked
    '''
    # assert type(color) == int, "color should be 0(black)/1(white)"
    # assert len(w_castling) == 2, "w_castling should be length 2 list, [1/0 kingside, 1/0 queenside]"
    # assert len(b_castling) == 2, "b_castling should be length 2 list, [1/0 kingside, 1/0 queenside]"

    board_shape = boards[0].shape
    layers = []
    if len(boards) < TIMESTEPS:
        num_zeros = int(TIMESTEPS - boards)
        zero_layers = np.zeros((num_zeros, board_shape[0], board_shape[1]))
        layers.append(zero_layers)

    id_boards = chess_env.transform_boards(boards)

    for id_board in id_boards:
        ids = [v for v in chess_env.pieces_to_id.values() if v != 0]  # 0 corresponds to '.', which is not a piece
        for id in ids:
            mask = np.ones_like(id_board) * int(id)
            bin_board = mask == id_board
            layers.append(bin_board)

    color_layer = np.ones(board_shape) if color == chess.WHITE else np.zeros(board_shape)
    layers.append(color_layer)

    moves_layer = np.ones(board_shape) * total_moves
    layers.append(moves_layer)

    w_castling_layers = [np.ones(board_shape) * x for x in w_castling]
    layers.extend(w_castling_layers)

    b_castling_layers = [np.ones(board_shape) * x for x in b_castling]
    layers.extend(b_castling_layers)

    no_progress_layer = np.ones(board_shape) * no_progress_count
    layers.append(no_progress_layer)

    return torch.cat(layers)


def get_additional_feature(board: chess.Board):
    '''
    Get additional features like color, total move count, etc from a board state
    :param board: chess.Board
    :return: additional features
    '''

    color = board.turn
    total_moves = board.fullmove_number
    w_castling = [board.has_kingside_castling_rights(chess.WHITE),
                  board.has_queenside_castling_rights(chess.WHITE)]
    b_castling = [board.has_kingside_castling_rights(chess.BLACK),
                  board.has_queenside_castling_rights(chess.BLACK)]
    no_progress_count = int(board.halfmove_clock / 2)

    return color, total_moves, w_castling, b_castling, no_progress_count


def learn(model, optimizer, dataloader, env):
    for _, samples in enumerate(dataloader):
        s, pi, z = samples[:, 0], samples[:, 1], samples[:, 2]
        extra_features = get_additional_feature(s)  # TODO: Make sure this works as intended
        planes = board_to_layers(*extra_features)
        p, v = model(torch.FloatTensor(planes))

        optimizer.zero_grad()
        value_loss = nn.MSELoss()(v, z)
        policy_loss = nn.CrossEntropyLoss()(p, pi)
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()
