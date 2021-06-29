import torch
import chess
import torch.nn as nn
import numpy as np

import chess_env
import utils

from alphazero.mcts.mcts import MCTS, ChessBoard
from utils import TBLogger, log
from chess_utils import encode_action


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TIMESTEPS = 8
M = 13  # From AlphaZero: we have NxN (MT + L) layers. In the paper, M=14 cuz of repetitions=2 instead of 1 for whatever reason
SIMULATIONS = 3
logger = TBLogger()


def board_to_layers(boards, repetitions, color, total_moves, w_castling, b_castling, no_progress_count):
    '''
    convert chess.Board to binary layers to feed into a neural network.
    :param env: ChessEnv
    :param boards: chess.Board game states
    :param repetitions: number of repetitions of the latest game state
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
    id_boards = chess_env.transform_boards(boards)
    board_shape = id_boards[0].shape
    layers = []
    if len(boards) < TIMESTEPS:
        num_zeros = int(TIMESTEPS - len(boards))
        zero_layers = torch.zeros((num_zeros * M, board_shape[0], board_shape[1]))
        layers.extend([zl for zl in zero_layers])

    for id_board in id_boards:
        id_board = torch.Tensor(id_board)
        reps_layer = torch.ones_like(id_board) * repetitions
        layers.append(reps_layer)
        ids = [v for v in chess_env.pieces_to_id.values() if v != 0]  # 0 corresponds to '.', which is not a piece
        for id in ids:
            mask = torch.ones_like(id_board) * int(id)
            bin_board = torch.eq(mask, id_board).to(torch.float32)
            layers.append(bin_board)

    color_layer = torch.ones(board_shape) if color == chess.WHITE else torch.zeros(board_shape)
    layers.append(color_layer)

    moves_layer = torch.ones(board_shape) * total_moves
    layers.append(moves_layer)

    w_castling_layers = [torch.ones(board_shape) * x for x in w_castling]
    layers.extend(w_castling_layers)

    b_castling_layers = [torch.ones(board_shape) * x for x in b_castling]
    layers.extend(b_castling_layers)

    no_progress_layer = torch.ones(board_shape) * no_progress_count
    layers.append(no_progress_layer)

    return torch.cat(layers).view(1, -1, board_shape[0], board_shape[1])


def get_additional_features(board: chess.Board):
    '''
    Get additional features like color, total move count, etc from a board state
    :param board: chess.Board
    :param env: ChessEnv environment
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


def self_play(root, model, env, queue):
    state = ChessBoard(root, None)
    data = []  # stores (s_t, pi_t, z_t)
    done = False
    new_game = True
    while not done:
        if new_game:
            mcts = MCTS(state, model)
            new_game = False
        for i in range(SIMULATIONS):
            mcts.search()
            # log.debug(f'Finished simulation {i}')
        action, new_root, entry = mcts.play()
        data.append(entry)
        new_root.delete_parent()  # delete tree above the new root
        log.info(f'New root: \n {new_root.board} \n fen: {new_root.board.fen()}')
        obs, rew, done, info = env.step(chess.Move.from_uci(action))
        mcts.root = new_root
    for entry in data:
        # retroactively apply rewards now that we know the terminal reward
        if info['winner'] is None:
            # draw
            entry[-1] = torch.Tensor([rew])
        else:
            entry[-1] = torch.Tensor([rew]) if entry[0].board.turn == info['winner'] else torch.Tensor([-rew])
    log.info(f'Reward for game: {rew}')
    info['rew'] = rew
    info['game_length'] = int(mcts.root.board.fen().split(' ')[-1])
    return data, info


def learn(model, optimizer, dataloader, env):
    total_loss = 0
    for i, samples in enumerate(dataloader):
        states, pi, z = samples[0], samples[1], samples[2]
        pi = pi.to(device)
        z = z.to(device)
        batch_size = pi.shape[0]
        extra_features = [get_additional_features(s.board) for s in states]
        planes = torch.stack([board_to_layers(s.board_stack, s.repetitions, *extra_feature) for s, extra_feature in zip(states, extra_features)]).squeeze(1).to(device)
        p, v = model(planes)
        if batch_size == 1:
            p = p.view(1, -1)

        np_boards = [np.array(str(s.board).split()).reshape(8, 8) for s in states]
        action_inds = [[encode_action(np_board, move, flattened=True) for move in s.legal_moves] for np_board, s in zip(np_boards, states)]
        p_a = [p[i, acts] for i, acts in enumerate(action_inds)]
        p_a = nn.utils.rnn.pad_sequence(p_a).T.to(device)
        logp = torch.log(p_a)

        optimizer.zero_grad()
        value_loss = nn.MSELoss()(v, z)
        policy_loss = -torch.mean(torch.sum(pi * logp, dim=1))
        loss = value_loss + policy_loss
        total_loss += loss
        loss.backward()
        optimizer.step()

    return total_loss
