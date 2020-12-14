import torch
import chess
import torch.optim as optim
import numpy as np

import chess_env

from alphazero.model import AlphaZero
from self_play import self_play

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOTAL_STEPS = 7e5


def board_to_layers(env, boards, color, total_moves, w_castling, b_castling, no_progress_count):
    '''
    convert chess.Board to binary layers to feed into a neural network
    :param env: ChessEnv
    :param boards: chess.Board game state
    :param color: white's turn (1) or black's turn (0)
    :param total_moves: total moves elapsed in this game
    :param w_castling: legality of white castling kingside or queenside
    :param b_castling: legality of black castling kingside or queenside
    :param no_progress_count: number of moves without progress (50 moves without progress is automatic draw)
    :return: torch Tensor of all layers stacked
    '''
    assert type(color) == int, "color should be 0(black)/1(white)"
    assert len(w_castling) == 2, "w_castling should be length 2 list, [1/0 kingside, 1/0 queenside]"
    assert len(b_castling) == 2, "b_castling should be length 2 list, [1/0 kingside, 1/0 queenside]"

    layers = []
    for board in boards:
        id_board = env.transform_board(board)
        ids = [v for v in chess_env.pieces_to_id.values() if v != 0]  # 0 corresponds to '.', which is not a piece
        for id in ids:
            mask = np.ones_like(id_board) * int(id)
            bin_board = mask == id_board
            layers.append(bin_board)

    color_layer = np.ones(env.observation_space.shape) if color == chess.WHITE else np.zeros(env.observation_space.shape)
    layers.append(color_layer)

    moves_layer = np.ones(env.observation_space.shape) * total_moves
    layers.append(moves_layer)

    w_castling_layers = [np.ones(env.observation_space.shape) * x for x in w_castling]
    layers.extend(w_castling_layers)

    b_castling_layers = [np.ones(env.observation_space.shape) * x for x in b_castling]
    layers.extend(b_castling_layers)

    no_progress_layer = np.ones(env.observation_space.shape) * no_progress_count
    layers.append(no_progress_layer)

    return torch.cat(layers)


def learn(model: AlphaZero, data, env):
    obs_dim = int(env.observation_space.shape[0] * env.observation_space.shape[1])
    n_acts = env.action.space.n
    model = AlphaZero(env.n_planes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.2)

    obs = torch.FloatTensor(env.reset()).to(device)
    step = 0
    while step < TOTAL_STEPS:
        pass 