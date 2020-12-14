import gym
import chess
import numpy as np

from gym import spaces
from chess import svg

#  capitalized = White
pieces_to_id = {
    'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5, 'P': 6,
    'r': -1, 'n': -2, 'b': -3, 'q': -4, 'k': -5, 'p': -6,
    '.': 0
}


class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.board_render = svg.board()

        self.n_discrete_actions = 4672  # all possible 'queen' and 'knight' moves from an arbitrary square to any other arbitrary square
        self.action_space = spaces.Discrete(self.n_discrete_actions)
        self.steps = 8
        self.observation_space = gym.spaces.Box(-6, 6, (8, 8))
        self.n_planes = 119  # from AlphaZero paper

    def transform_board(self, board):
        '''
        convert chess.Board representation to numeric id board representation
        :param board: chess.Board
        :return: id_board numerical representation
        '''
        board = str(board).split()
        board = [pieces_to_id[s] for s in board]
        board = np.array(board).reshape((8, 8))
        return board

    def step(self, action: chess.Move):
        '''
        :param action: chess.Move format
        :return: obs, rew, dones, info
        '''
        assert type(action) == chess.Move

        if action not in self.board.legal_moves:
            raise ValueError(
                f'Illegal move {action} passed to step function for board position {self.board.fen()}'
            )

        self.board.push(action)

        obs = self.board.copy()
        if self.board.turn == chess.BLACK:
            obs = self.board.transform(chess.flip_vertical).copy()
        obs = str(obs).split()
        obs = [pieces_to_id[s] for s in obs]
        obs = np.array(obs).reshape(self.observation_space.shape)

        rew = self.board.result()
        done = self.board.is_game_over()
        info = None

        return obs, rew, done, info

    def reset(self):
        self.board.reset()
        obs = self.board.copy()
        obs = str(obs).split()
        obs = [pieces_to_id[s] for s in obs]
        obs = np.array(obs).reshape(self.observation_space.shape)
        return obs

    def render(self, mode='human'):
        print(self.board)

    def close(self):
        pass
