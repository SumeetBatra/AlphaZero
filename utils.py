import os
import torch
import torch.optim as optim
import logging
import logging.handlers
import multiprocessing
import time
from random import random, randint

from torch.utils.tensorboard import SummaryWriter
from colorlog import ColoredFormatter

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s][%(processName)-10s] %(message)s",
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

if not os.path.isdir('logs'):
    os.makedirs('logs')
if not os.path.isfile('logs/log.txt'):
    file = open('logs/log.txt', 'w+')

fh = logging.FileHandler('logs/log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

log = logging.getLogger('rl')
log.setLevel(logging.DEBUG)
log.handlers = []  # No duplicated handlers
log.propagate = False  # workaround for duplicated logs in ipython
log.addHandler(ch)
log.addHandler(fh)

# Because you'll want to define the logging configurations for listener and workers, the
# listener and worker process functions take a configurer parameter which is a callable
# for configuring logging for that process. These functions are also passed the queue,
# which they use for communication.
def listener_configurer():
    log = logging.getLogger('rl')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    fh = logging.FileHandler('logs/log.txt')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(ch)
    log.addHandler(fh)

# This is the listener process top-level loop: wait for logging events
# (LogRecords)on the queue and handle them, quit when you get a None for a
# LogRecord.
def listener_process(queue, configurer):
    configurer()
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

# The worker configuration is done at the start of the worker process run.
# Note that on Windows you can't rely on fork semantics, so each process
# will run the logging configuration code when it starts.
def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.DEBUG)

# This is the worker process top-level loop, which just logs ten events with
# random intervening delays before terminating.
# The print messages are just so you know it's doing something!
def worker_process(queue, configurer, func):
    configurer(queue)
    name = multiprocessing.current_process().name
    func()
    print('Worker finished: %s' % name)


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
