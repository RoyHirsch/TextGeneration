import logging
import os
import time
import numpy as np
import torch
import random
import json
from matplotlib import pyplot as plt

from datetime import timedelta

def set_initial_random_seed(random_seed):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

def save_tensor(x, file_name):
    torch.save(x, file_name)

def dump_to_json(dict, name_str, dir=None):
    file = json.dumps(dict)
    if dir:
        file_name = os.path.join(dir, name_str + ".json")
    else:
        file_name = name_str + ".json"

    f = open(file_name, "w")
    f.write(file)
    f.close()

def read_text_file_to_list(f_name):
    with open(f_name, 'r') as f:
        data = f.readlines()
    return data

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, test_acc):

        score = test_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)

def create_logger(log_dir, dump=True):
    filepath = os.path.join(log_dir, 'net_launcher_log.log')

    if not os.path.exists(log_dir) and log_dir:
        os.makedirs(log_dir)

    # Create logger
    log_formatter = LogFormatter()

    if dump:
        # create file handler and set level to info
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to info
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if dump:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info('Created logger')
    return logger

