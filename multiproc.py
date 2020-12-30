import torch.multiprocessing as mp


class Actor:
    def __init__(self, target, obs, model, env, q):
        self.target = target
        self.obs = obs
        self.model = model
        self.env = env
        self.shared_queue = q
        self.terminated = False

    @property
    def is_terminated(self):
        return self.terminated

    def run_game(self):
        p = mp.Process(target=self.target, args=(self.obs, self.model, self.env, self.shared_queue))
        p.start()


def actor(target, *args):
    p = mp.Process(target=target, args=args)
    p.start()


def evaluator(target, *args):
    p = mp.Process(target=target, args=args)
    p.start()
