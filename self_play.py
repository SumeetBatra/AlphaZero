from alphazero.mcts import MCTS
from alphazero.model import AlphaZero
from chess_env import ChessEnv
from chess import Board



SIMULATIONS = 800


def self_play(state: Board, model: AlphaZero):
    mcts = MCTS(state)
    for i in range(SIMULATIONS):
        mcts.search(model)
    data = mcts.play()
    return data