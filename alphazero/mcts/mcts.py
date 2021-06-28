import learn
import time
import torch.multiprocessing as mp

from abc import ABC, abstractmethod
from torch.distributions.categorical import Categorical
from chess_utils import *
from utils import log

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TIME_STEPS = 8


class MCTS(object):
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self._encoded_state_counter = dict()
        self.root.expand(model, root=True)

    @property
    def root_node(self):
        return self.root

    @root_node.setter
    def root_node(self, new_root):
        self.root = new_root

    def _encode_state(self, board: chess.Board):
        '''
        Encode the board to check for repetitions and return repetitions
        :param board: chess.Board
        '''
        encoding = board.fen()
        self._encoded_state_counter[encoding] = self._encoded_state_counter.get(encoding, 0) + 1

        return self._encoded_state_counter[encoding]

    def _reset_internal_state(self):
        '''
        reset the encoded state counter and max repetitions for a new search
        '''
        self._encoded_state_counter.clear()

    def search(self):
        leaf = self.root.select_leaf()
        val = leaf.expand(self.model)
        self._backprop(leaf, val)

    @staticmethod
    def _backprop(leaf, val):
        leaf.visit_count += 1
        node = leaf
        while node.parent:
            # node.visit_count += 1
            node.parent.child_W[node.move] += val + 1  # +1 to counteract virtual loss
            node.parent.child_Q[node.move] = node.parent.child_W[node.move] / node.parent.child_N[node.move]
            node = node.parent
            val = -val

    def _sample(self, temp):
        '''
        sample a move based on the exponentiated visit count
        :param temp: temperature hyper-parameter
        :return: child node
        '''
        logits = self.root.child_N / self.root.visit_count
        if temp == 0:
            # greedy select the best action
            action = np.argmax(logits)
            logits = np.zeros_like(logits)
            logits[action] = 1.0  # need this to parametrize the policy
        else:
            logits = np.power(np.array(logits), 1.0 / temp)
            action = np.random.choice(len(self.root.legal_moves), p=logits)
        return self.root.legal_moves[action], self.root.children[action], [self.root.board, Categorical(torch.Tensor(logits)), None]  # we don't know the reward yet - will be retroactively updated

    def play(self):
        action, child, datapoint = self._sample(temp=1.0)
        return action, child, datapoint


class MCTSNode(ABC):

    @abstractmethod
    def expand(self, model, root=False):
        pass

    @abstractmethod
    def select_leaf(self):
        pass

    @abstractmethod
    def best_child(self):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def puct(self, cpuc=1.0):
        """UCT upper confidence score variant used in AlphaZero"""
        pass


class ChessBoard(MCTSNode):
    def __init__(self, board, move, parent=None):
        super(ChessBoard).__init__()
        self.parent = parent
        self.move = move  # idx of move that was taken to get to this node. None if root
        self.board = board
        self.color = board.turn
        self.visits = 0 # only used for root node b/c it has no parent
        if self.color == chess.BLACK:
            #  board should be in orientation of current player when fed to the neural network
            nn_board = board.copy()
            nn_board.apply_transform(chess.flip_vertical)
            nn_board.apply_transform(chess.flip_horizontal)
        else:
            nn_board = board
        self.board_stack = [nn_board]
        if self.parent:
            self.board_stack = self.parent.board_stack[-7:] + self.board_stack
        self.legal_moves = [move.uci() for move in board.legal_moves]
        self.visited = False
        self.repetitions = 0
        self.children = {}  # {move idx: MCTSNode}
        self.child_N = np.zeros(len(self.legal_moves), dtype=np.float32)
        self.child_W = np.zeros(len(self.legal_moves), dtype=np.float32)
        self.child_Q = np.zeros(len(self.legal_moves), dtype=np.float32)
        self.child_P = np.zeros(len(self.legal_moves), dtype=np.float32)

    @property
    def visit_count(self):
        return self.parent.child_N[self.move] if self.parent else self.visits  # if self.parent == None i.e root node, return 1

    @visit_count.setter
    def visit_count(self, visits):
        if self.parent:
            self.parent.child_N[self.move] = visits
        else:
            self.visits = visits


    @property
    def total_value(self):
        return self.parent.child_W[self.move] if self.parent else 1

    @total_value.setter
    def total_value(self, value):
        if self.parent: self.parent.child_W[self.move] = value

    def puct(self, cpuct=3.0, eps=1e-8):
        '''
        Vectorized PUCT algorithm.
        :param cpuct: Hyperparameter controlling exploration
        :param eps: epsilon to be added inside the square root to prevent all zeros
        :return: PUCT Score
        '''
        # see https://ai.stackexchange.com/questions/25451/how-does-alphazeros-mcts-work-when-starting-from-the-root-node for why U is calculated this way
        U = cpuct * self.child_P * np.sqrt(self.visit_count + eps) / (1.0 + self.child_N)
        return self.child_Q + U

    def best_child(self):
        move_idx = np.argmax(self.puct())
        child = self.children.get(move_idx, None)
        if child:
            return child
        else:
            new_board = self.board.copy()
            new_board.push(chess.Move.from_uci(self.legal_moves[move_idx]))
            child = ChessBoard(new_board, move_idx, self)
            self.children[move_idx] = child
            return child

    def select_leaf(self):
        node = self
        while node.visited:
            ##############################################
            # virtual loss
            node.visit_count = node.visit_count + 1
            node.total_value -= 1
            ##############################################
            node = node.best_child()

            # log.debug(f'Best child: \n {node.board}\n')
            # time.sleep(0.1)
        return node

    def expand(self, model, root=False):
        self.visited = True
        p_acts, val = self.action_value(model)
        if root:  # add dirichlet noise to root node for exploration
            eps = 0.25
            alpha = torch.ones_like(p_acts) * 0.3
            noise = torch.distributions.dirichlet.Dirichlet(alpha).sample()
            p_acts = (1-eps) * p_acts + eps * noise
        np_board = np.array(str(self.board).split()).reshape(8, 8)
        action_inds = [encode_action(np_board, move, flattened=True) for move in self.legal_moves]
        p_a = p_acts[action_inds].detach().numpy()
        p_a = p_a / np.sum(p_a)  # renormalize the probabilities
        self.child_P = p_a
        return val

    def action_value(self, model):
        features = learn.get_additional_features(self.board)
        planes = learn.board_to_layers(self.board_stack, self.repetitions, *features)
        p_acts, val = model(torch.FloatTensor(planes).to(device))
        return p_acts, val

    def is_terminal(self):
        return self.board.is_game_over() or self.repetitions >= 3

    def print(self):
        # easy to visualize the board during debugging
        return np.array(str(self.board).split()).reshape(8, 8)

    def delete_parent(self):
        # delete parent so that current node becomes the new root
        self.visits = self.visit_count - 1  # subtract 1 so that root visits = sum(child_N)
        self.parent = None
