import sys
sys.path.insert(0, '/data/sumeet/AlphaZero')
import chess_env
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from model import AlphaZero
from learn import get_additional_features, board_to_layers


def save_torchscript():
    env = chess_env.ChessEnv()
    model = AlphaZero(env.n_planes)
    board = [env.board]
    features = get_additional_features(board[0])
    repetitions = 0
    planes = board_to_layers(board, repetitions, *features)

    sm = torch.jit.trace(model, planes)
    sm.save("annotated_alphazero.pt")

if __name__ == '__main__':
    save_torchscript()