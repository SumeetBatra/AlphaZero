import time
import random
import chess
import torch
import numpy as np
import torch.optim as optim
import torch.multiprocessing as multiprocessing

import chess_env

from learn import *
from chess_utils import *
from alphazero.model import AlphaZero

from torch.utils.data.dataloader import DataLoader
from torch.distributions.categorical import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TOTAL_STEPS = int(7e5)
USE_TENSORBOARD = False
BATCH_SIZE = 32
L2_REG = 1e-4


def test():
    '''
    Code for testing things out
    '''
    env = chess_env.ChessEnv()
    board = env.board
    # print("legal: ", [mv.uci() for mv in board.legal_moves])
    # move = chess.Move.from_uci('b2b4')
    # print(move.from_square, move.to_square)
    # str_board = np.array(str(board).split()).reshape(8, 8)
    # print(str_board)
    # # board.push(move)
    # # print(board.transform(chess.flip_vertical).copy())
    # print(board)
    # print('\n')
    # # print(board.copy())
    # vis1 = str(board).split()
    # # print("last: ", vis, len(vis))
    # vis2 = np.array(vis1).reshape(8, 8)
    # print(vis1)
    # env.step(move)
    # print(board)
    # print('\n')
    # print(board.transform(chess.flip_vertical))
    print(board)
    print('\n')
    board.apply_transform(chess.flip_vertical)
    board.apply_transform(chess.flip_horizontal)
    print(board)
    print('\n')
    model = AlphaZero(env.n_planes).to(device)
    board.clear_board()
    board.set_piece_at(chess.A7, chess.Piece(chess.PAWN, chess.WHITE))
    # move = chess.Move.from_uci('a7a8q')
    # board.push(move)
    print(board)
    features = get_additional_features(board, env)
    layers = board_to_layers([board], *features)
    act_probs, val = model(layers)
    filtered_act_probs = mask_illegal_actions(board, act_probs).view(-1)
    filtered_acts = torch.nonzero(filtered_act_probs)
    acts = []
    for act in filtered_acts:
        acts.append(decode_action(act, board))
    print(all(e in acts for e in list(board.legal_moves)))


def play_random_game(model, board, env):
    t = 0
    while not board.is_game_over():
        print(t)
        features = get_additional_features(board, env)
        layers = board_to_layers([board], *features)
        act_probs, val = model(layers)
        filtered_act_probs = mask_illegal_actions(board, act_probs).view(-1)
        nonzero_filtered_act_probs = torch.nonzero(filtered_act_probs).detach().numpy()
        filtered_acts = []
        for act in nonzero_filtered_act_probs:
            decoded = decode_action(int(act), board)
            filtered_acts.append(decoded)
        action = str(random_action(filtered_acts, list(board.legal_moves)))
        move = chess.Move.from_uci(action)
        board.push(move)
        t += 1


def random_action(filtered_acts, legal_moves):
    assert all(e in filtered_acts for e in legal_moves), f'Filtered actions: {filtered_acts} and legal moves: {legal_moves} are not the same :('
    return np.random.choice(filtered_acts)


def train():
    env = chess_env.ChessEnv()
    obs_dim = int(env.observation_space.shape[0] * env.observation_space.shape[1])
    n_acts = env.action_space.n
    model = AlphaZero(env.n_planes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.2, weight_decay=L2_REG)

    obs = env.reset()

    for i in range(TOTAL_STEPS):
        if i > 0:
            data = self_play(obs, model, env)
            train_data = ChessDataset(data)
            dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
            learn(env, model, optimizer, dataloader)
        else:
            data = self_play(obs, model, env)
            train_data = ChessDataset(data)
            dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    # test()
    train()
    # env = chess_env.ChessEnv()
    # board = env.board
    # model = AlphaZero(env.n_planes).to(device)
    # play_random_game(model, board, env)
