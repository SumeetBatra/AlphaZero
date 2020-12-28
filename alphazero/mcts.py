import learn

from abc import ABC, abstractmethod
from torch.distributions.categorical import Categorical
from chess_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TIME_STEPS = 8


class MCTS:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self._encoded_state_counter = dict()
        self.root.expand(model, root=True)

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

    def _select(self):
        """
        Rollout a game using current statistics. Store new unvisited leaf on the tree
        :return: Leaf node
        """
        node = self.root
        acts = []
        while True:
            child, taken_act = node.take_best_action()
            child.repetitions = self._encode_state(child.board)
            acts.append(taken_act)
            if child.is_terminal():
                return child, acts
            if not child.visited:
                child.visited = True

                node.children[taken_act] = child
                return child, acts
            node = child

    def _expand(self, leaf):
        """
        Send the leaf node to the neural network for evaluation
        :return: value v and the leaf node
        """
        return leaf.expand(self.model)

    @staticmethod
    def _backprop(node, val, acts):
        for act in list(reversed(acts)):
            n, w, _, p = node.parent.actions[act]
            n += 1
            w += val
            q = w / n
            node.parent.actions[act] = (n, w, q, p)
            node = node.parent

    @staticmethod
    def _sample(node, temp):
        '''
        sample a move based on exponentiated visit count
        :return: child node based on the move that was sampled
        '''
        logits = []
        actions, stats = list(node.actions.keys()), list(node.actions.values())
        n_total = sum(list([n for n, _, _, _ in node.actions.values()]))
        for stat in stats:
            n, w, q, p = stat
            logits.append(n / n_total)
        if temp == 0:
            # greedy select the best action
            max_idx = np.argmax(logits)
            logits = np.zeros(len(logits))
            logits[max_idx] = 1.0
        else:
            logits = np.power(np.array(logits), 1 / temp)
        action = np.random.choice(actions, p=logits)
        return chess.Move.from_uci(action), node.children[action], [node, Categorical(torch.Tensor(logits)), None]  # we don't know the reward yet - will be retroactively updated

    def search(self):
        self._reset_internal_state()
        leaf, acts = self._select()
        val = self._expand(leaf)
        self._backprop(leaf, val, acts)

    def play(self):
        node = self.root
        action, child, data_entry = self._sample(node, temp=1.0)
        return action, child, data_entry


class MCTSNode(ABC):

    @abstractmethod
    def expand(self, model, root=False):
        pass

    @abstractmethod
    def take_best_action(self):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def puct(self, n, n_total, p, q, coeff=1.0):
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
        self.visited = False
        self.repetitions = 0

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

    def action_value(self, model):
        features = learn.get_additional_features(self.board)
        boards = self.board_stack[-7:] + [self.board]
        planes = learn.board_to_layers(boards, self.repetitions, *features)
        p_acts, val = model(torch.FloatTensor(planes).to(device))
        return p_acts, val

    def expand(self, model, root=False):
        if self.is_terminal():
            return self.terminal_reward()
        p_acts, val = self.action_value(model)
        if root:  # add dirichlet noise to root node for exploration
            eps = 0.25
            alpha = torch.ones_like(p_acts) * 0.3
            noise = torch.distributions.dirichlet.Dirichlet(alpha).sample()
            p_acts = (1-eps) * p_acts + eps * noise
        for move in self.legal_moves:
            a_idx = encode_action(np.array(str(self.board).split()).reshape(8, 8), move.uci(), flattened=True)
            p_a = p_acts.squeeze()[a_idx]
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
            n_total = sum(list([n for n, _, _, _ in self.actions.values()]))
            score = self.puct(n, n_total, p, q)
            if score > max_score:
                max_score = score
                best_act = act
        new_board = self.board.copy()
        new_board.push(chess.Move.from_uci(best_act))
        child = self.children.get(best_act, ChessBoard(new_board, self))
        return child, best_act

    def puct(self, n, n_total, p, q, coeff=1.0):
        Q = q
        U = coeff * p * np.sqrt(n_total) / (1 + n)
        return Q + U

    def is_terminal(self):
        return self.board.is_game_over() or self.repetitions >= 3

