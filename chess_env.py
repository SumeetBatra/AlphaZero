import gym
import chess

from gym import spaces
from chess import svg

class Chess(gym.Env):
	def __init__(self):
		super(Chess, self).__init__()
		self.board = chess.Board()
		self.board_render = svg.board()

		self.n_discrete_actions = 4672 # all possible 'queen' and 'knight' moves from an arbitrary square to any other arbitrary square
		self.action_space = spaces.Discrete(self.n_discrete_actions)
		self.steps = 8
		self.observation_space = None # TODO: fix this???

	def step(self, action: chess.Move):
		'''
		:param action: chess.Move format
		:return: obs, rew, dones, info
		'''
		assert type(action) == chess.Move

		if action not in self.board.legal_moves:
			raise ValueError (
				f'Illegal move {action} passed to step function for board position {self.board.fen()}'
			)

		self.board.push(action)

		obs = self.board.copy()
		if self.board.turn == chess.BLACK:
			obs = self.board.transform(chess.flip_vertical).copy()

		rew = self.board.result()
		done = self.board.is_game_over()
		info = None

		return obs, rew, done, info


	def reset(self):
		self.board.reset()

	def render(self, mode='human'):
		print(self.board)

	def close(self):
		pass