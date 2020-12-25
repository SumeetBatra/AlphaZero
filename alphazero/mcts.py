import chess
import torch
import numpy as np

import learn

from abc import ABC, abstractmethod
from torch.distributions.categorical import Categorical
from chess_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TIME_STEPS = 8


class MCTS:
    def __init__(self, state: chess.Board, env):
        self.root = ChessBoard(state, 0)
        self.data = []  # stores (s_t, pi_t, z_t)
        self.env = env

    def _select(self):
        """
        Rollout a game using current statistics. Store new unvisited leaf on the tree
        :return: Leaf node
        """
        node = self.root
        acts = []
        while True:
            child, taken_act = node.take_best_action()
            self.env.step(chess.Move.from_uci(taken_act))
            acts.append(taken_act)
            if child.is_terminal():
                return child, acts
            if not child.visited:
                child.visited = True
                node.children[taken_act] = child
                return child, acts
            node = child

    def _expand(self, leaf, model):
        """
        Send the leaf node to the neural network for evaluation
        :return: value v and the leaf node
        """
        return leaf.expand(model, self.env)

    @staticmethod
    def _backprop(node, val, acts):
        for act in list(reversed(acts)):
            n, w, _, p = node.parent.actions[act]
            n += 1
            w += val
            q = w / n
            node.parent.actions[act] = (n, w, q, p)
            node = node.parent

    def _sample(self, node, temp):
        '''
        sample a move based on exponentiated visit count
        :return: child node based on the move that was sampled
        '''
        logits = []
        actions, stats = list(node.actions.keys()), list(node.actions.values())
        n_total = sum([n for n, _, _, _ in node.actions.values()])
        for stat in stats:
            n, w, q, p = stat
            logits.appends(n / n_total)
        if temp == 0:
            # greedy select the best action
            max_idx = np.argmax(logits)
            logits = np.zeros(len(logits))
            logits[max_idx] = 1.0
        else:
            logits = np.power(np.array(logits), 1 / temp)
        self.data.append([node, Categorical(torch.Tensor(logits)), None])  # we don't know the reward yet - will be retroactively updated
        action = np.random.choice(actions, p=logits)
        return self.children[action]


        logits = []
        moves, children = list(node.children.keys()), list(node.children.values())
        for child in children:
            logits.append(child.N / node.N_total)
        if temp == 0:
            # greedy select the best action
            max_idx = np.argmax(logits)
            logits = np.zeros(len(logits))
            logits[max_idx] = 1.0
        else:
            logits = np.power(np.array(logits), 1 / temp)
        self.data.append([node, Categorical(torch.Tensor(logits)), None])  # we don't know the reward yet - will be retroactively updated
        child = np.random.choice(children, p=logits)
        return child

    def search(self, model, env):
        _ = env.reset()
        leaf, acts = self._select()
        val = self._expand(leaf, model)
        self._backprop(leaf, val, acts)

    def play(self):
        node = self.root
        while not node.is_terminal():
            node = self._sample(node, temp=1.0)
        reward = node.terminal_reward()
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
    def __init__(self, board, parent=None):
        super(ChessBoard).__init__()
        self.board = board
        self.board_stack = [board]
        self.color = board.turn
        self.parent = parent
        self.legal_moves = list(self.board.legal_moves)
        self.children = dict()
        self.N_total = 0
        self.visited = False

        if self.parent:
            self.board_stack.extend(self.parent.board_stack)

        self.actions = dict()

    def terminal_reward(self):
        res = self.board.result()[0:2]  # can be '1-0', '0-1', or '1/2-1/2'
        if res == '1-':  # current player wins
            return 1.0
        elif res == '0-':  # opponent wins
            return -1.0
        elif res == '1/':  # draw
            return 0.0

    def action_value(self, model, env):
        features = learn.get_additional_features(self.board, env)
        boards = self.board_stack[-7:] + [self.board]
        planes = learn.board_to_layers(boards, *features)
        p_acts, val = model(torch.FloatTensor(planes).to(device))
        return p_acts, val

    def expand(self, model, env):
        if self.is_terminal():
            return self.terminal_reward()
        p_acts, val = self.action_value(model, env)
        for move in self.legal_moves:
            a_idx = encode_action(np.array(str(self.board)).reshape(8, 8), move.uci(), flattened=True)
            p_a = p_acts[a_idx]
            self.actions[move.uci()] = (0, 0, 0, p_a)
        return val

    def take_best_action(self):
        if self.is_terminal():
            return self
        assert len(self.actions) != 0, "This node has not been expanded yet"
        best_act = None
        max_score = -1
        for act, stats in self.actions.items():
            n, w, q, p = stats
            score = self.puct(n, p, q)
            if score > max_score:
                max_score = score
                best_act = act
        child = self.children.get(best_act, ChessBoard(self.board.copy().push(chess.Move.from_uci(best_act)), self))
        return child, best_act

    def puct(self, n, p, q, coeff=1.0):
        Q = q
        U = coeff * p * np.sqrt(self.N_total) / (1 + n)
        return Q + U

    def is_terminal(self):
        return self.board.is_game_over()

