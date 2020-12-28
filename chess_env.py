import gym
import chess
import numpy as np

from gym import spaces

#  capitalized = White
pieces_to_id = {
    'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5, 'P': 6,
    'r': -1, 'n': -2, 'b': -3, 'q': -4, 'k': -5, 'p': -6,
    '.': 0
}


def transform_boards(boards):
    '''
    convert chess.Board(s) representation to numerical id board(s) representation
    :param board: chess.Board(s)
    :return: id_board(s) numerical representation
    '''
    boards = [str(board).split() for board in boards]
    boards = [[pieces_to_id[s] for s in board] for board in boards]
    boards = [np.array(board).reshape((8, 8)) for board in boards]
    return boards


class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self._encoded_state_counter = dict()
        self._max_repetitions = 0
        self.n_discrete_actions = 4672  # all possible 'queen' and 'knight' moves from an arbitrary square to any other arbitrary square
        self.action_space = spaces.Discrete(self.n_discrete_actions)
        self.steps = 8
        self.observation_space = gym.spaces.Box(-6, 6, (8, 8))
        self.n_planes = 111  # AlphaZero is 119, but here I use repetitions = 1 instead of 2 b/c idk why you need 2

    @property
    def repetitions(self):
        return self._max_repetitions

    @property
    def is_terminal(self):
        return self.board.is_game_over()

    def encode_state(self, board: chess.Board):
        '''
        Encode the board to check for repetitions
        :param board: chess.Board
        '''
        encoding = board.fen()
        self._encoded_state_counter[encoding] = self._encoded_state_counter.get(encoding, 0) + 1

        if self._encoded_state_counter[encoding] > self._max_repetitions:
            self._max_repetitions = self._encoded_state_counter[encoding]

    def step(self, action):
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
        self.encode_state(self.board)

        # flip the board to be in the current player's perspective
        board_copy = self.board.copy()
        board_copy.apply_transform(chess.flip_vertical)
        board_copy.apply_transform(chess.flip_horizontal)
        obs = board_copy

        rew = self.board.result()
        done = self.board.is_game_over() or self.repetitions >= 3
        info = None

        return obs, rew, done, info

    def reset(self):
        self.board.reset()
        obs = self.board.copy()
        self._max_repetitions = 0
        self._encoded_state_counter = dict()
        # obs = str(obs).split()
        # obs = [pieces_to_id[s] for s in obs]
        # obs = np.array(obs).reshape(self.observation_space.shape)
        return obs

    def render(self, mode='human'):
        print(self.board)

    def close(self):
        pass
