import os
import torch
import torch.optim as optim
import logging

from torch.utils.tensorboard import SummaryWriter
from colorlog import ColoredFormatter

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s][%(process)05d] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white,bold',
        'INFOV': 'cyan,bold',
        'WARNING': 'yellow',
        'ERROR': 'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('rl')
log.setLevel(logging.DEBUG)
log.handlers = []  # No duplicated handlers
log.propagate = False  # workaround for duplicated logs in ipython
log.addHandler(ch)



def save_checkpoint(cp_dir, cp_name, model, optimizer, **kwargs):
    os.makedirs(cp_dir, exist_ok=True)
    params = {}
    params['model_state_dict'] = model.state_dict()
    params['optim_state_dict'] = optimizer.state_dict()
    for key, val in kwargs:
        params[key] = val
    torch.save(params, cp_dir + cp_name)

def load_checkpoint(cp_path, model, optimizer):
    cp = torch.load(cp_path)
    model.load_state_dict(cp['model_state_dict'])
    optimizer.load_state_dict(cp['optim_state_dict'])
    return model, optimizer


class TBLogger():
    '''Tensorboard logger'''
    def __init__(self, logdir='./logs', port=6006):
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        self.port = port

        os.makedirs(self.logdir, exist_ok=True)

    def log(self, name, item, step):
        if isinstance(item, dict):
            self.writer.add_scalars(name, item, step)
        else:
            # item is a float or string/blobname
            self.writer.add_scalar(name, item, step)

    def grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)

    def close(self):
        self.writer.close()

