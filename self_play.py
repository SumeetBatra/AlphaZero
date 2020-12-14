from alphazero.mcts import MCTS
from alphazero.model import AlphaZero
from chess_env import ChessEnv
from chess import Board
from learn import learn



SIMULATIONS = 800


def self_play(state: Board, model: AlphaZero, env: ChessEnv):
    mcts = MCTS(state)
    for i in range(SIMULATIONS):
        mcts.search()
    data = mcts.play()
    learn(model, data, env)
