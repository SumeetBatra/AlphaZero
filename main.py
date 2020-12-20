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
    model = AlphaZero(env.n_planes).to(device)
    board.clear_board()
    board.set_piece_at(chess.A7, chess.Piece(chess.PAWN, chess.WHITE))
    # move = chess.Move.from_uci('a7a8q')
    # board.push(move)
    print(board)
    features = get_additional_features(board, env)
    layers = board_to_layers([board], *features)
    act_probs, val = model(layers)
    filtered_acts = mask_illegal_actions(board, act_probs)



def train():
    env = chess_env.ChessEnv()
    obs_dim = int(env.observation_space.shape[0] * env.observation_space.shape[1])
    n_acts = env.action_space.n
    model = AlphaZero(env.n_planes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.2, weight_decay=L2_REG)

    obs = env.reset()
    train_data = []
    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

    for i in range(TOTAL_STEPS):
        if i > 0:
            self_play(obs, model, env)
            learn(env, model, optimizer, dataloader)
        else:
            self_play(obs, model, env)


if __name__ == '__main__':
    test()
    # train()
