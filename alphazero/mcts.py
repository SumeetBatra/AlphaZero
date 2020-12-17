import chess
import torch
import numpy as np

import learn

from abc import ABC, abstractmethod
from torch.distributions.categorical import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TIME_STEPS = 8


class MCTS:
    def __init__(self, state: chess.Board, env):
        self.root = ChessBoard(state, 0)
        self.data = []  # stores (s_t, pi_t, z_t)
        self.env = env

    def _select(self):
        """
        Rollout a game using current statistics
        :return: Leaf node
        """
        node = self.root
        while node.children:
            child = node.take_best_action()
            node = child
        return node

    def _expand(self, leaf, model):
        """
        Send the leaf node to the neural network for evaluation
        :return: value v and the leaf node
        """
        return leaf.expand(model, self.env)

    @staticmethod
    def _backprop(node, val):
        while node:
            node.W += val
            node.N += 1
            node.Q = node.W / node.N
            node = node.parent

    def _sample(self, node, temp):
        '''
        sample a move based on exponentiated visit count
        :return: child node based on the move that was sampled
        '''
        logits = []
        for child in node.children:
            logits.append(child.N / node.N_total)
        if temp == 0:
            # greedy select the best action
            max_idx = np.argmax(logits)
            logits = np.zeros(len(logits))
            logits[max_idx] = 1.0
        else:
            logits = np.pow(np.array(logits), 1 / temp)
        self.data.append([node, Categorical(torch.Tensor(logits)), None])  # we don't know the reward yet - will be retroactively updated
        idx = np.random.choice(len(node.children), p=logits)
        return node.children[idx]


    def search(self, model):
        leaf = self._select()
        val = self._expand(leaf, model)
        self._backprop(leaf, val)

    def play(self):
        node = self.root
        while not node.is_terminal():
            node = self._sample(node, temp=1.0)
        reward = node.terminal_reward()  # TODO: add terminal state as data entry??
        # assign rewards
        for entry in self.data:
            entry[-1] = reward if node.color == entry[0].color else -reward
        return self.data



class MCTSNode(ABC):

    @abstractmethod
    def expand(self, model, env):
        pass

    @abstractmethod
    def take_best_action(self):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def puct(self, n, p, q, coeff=1.0):
        """UCT upper confidence score variant used in AlphaZero"""
        pass


class ChessBoard(MCTSNode):
    def __init__(self, board, p_a, parent=None):
        super(ChessBoard).__init__()
        self.board = board
        self.board_stack = [board]
        self.color = board.turn
        self.children = dict()  # {chess.move: ChessBoard Node} pair
        self.parent = parent
        self.untried_actions = list(self.board.legal_moves)
        self.N_total = 0  # sum of visit counts of all children
        # These are actually stats for the edge that led to this child node
        self.N = 0  # visit count to this node from parent
        self.W = 0
        self.Q = 0
        self.P = p_a

        if self.parent:
            self.board_stack.extend(self.parent.board_stack)

    def terminal_reward(self):
        res = self.board.result()[0:2]  # can be '1-0', '0-1', or '1/2-1/2'
        if res == '1-':  # current player wins
            return 1.0
        elif res == '0-':  # opponent wins
            return -1.0
        elif res == '1/':  # draw
            return 0.0

    def expand(self, model, env):
        if self.is_terminal():
            return self.terminal_reward()

        action = self.untried_actions.pop()
        new_board, rew, done, _ = env.step(action)

        boards = self.board_stack[-7:] + [new_board]
        color, total_moves, w_castling, b_castling, no_progress_count, repetitions = learn.get_additional_features(new_board, env)

        planes = learn.board_to_layers(boards, color, total_moves, w_castling, b_castling, no_progress_count, repetitions)
        p_a, val = model(torch.FloatTensor(planes).to(device))  # TODO: This should happen async. on another thread (?) AND should be stack of T boards from T prior timesteps
        child = ChessBoard(new_board, p_a, parent=self)
        self.children[child] = action
        return val

    def take_best_action(self):
        if not self.children:
            return self
        best_child = None
        best_score = -1
        for child in self.children.keys():
            score = self.puct(child.N, child.P, child.Q)
            if score > best_score:
                best_child = child
                best_score = score
        self.N_total += 1  # increment total visit count to children
        return best_child

    def is_terminal(self):
        return self.board.is_game_over()

    def puct(self, n, p, q, coeff=1.0):
        Q = q
        U = coeff * p * np.sqrt(self.N_total) / (1 + n)
        return Q + U



