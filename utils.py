import os
import json
import time
import random
import logging
import itertools
from datetime import datetime, timedelta
import numpy as np
import torch
from texttable import Texttable
torch.set_printoptions(threshold=float('inf'))


class SigmoidScheduler:
    def __init__(self, target_weight=1.0, center_step=2000, slope=0.005):
        """
        Args:
            target_weight: max w
            center_step: half w step
            slope: k.
        """
        self.target_weight = target_weight
        self.center_step = center_step
        self.slope = slope
        self.current_step = 0

    def step(self):
        self.current_step += 1
        return self.get_weight(self.current_step)

    def get_weight(self, step):
        # Sigmoid
        weight = self.target_weight / (1 + np.exp(-self.slope * (step - self.center_step)))
        return weight


def save_checkpoint(state_dict, epoch, args, filename="model.pt", optimizer=None, scheduler=None):
    save_dict = {"epoch": epoch, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    writer_path = os.path.join(args.logdir, 'session')
    filename = os.path.join(writer_path, filename)
    torch.save(save_dict, filename)
    return filename


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.2f}s"
    else:
        return f"{secs:.2f}s"


def calculate_eta(current_exp, total_experiments, experiment_times):
    if not experiment_times:
        return " | ETA: Calculating..."

    last_exp_time = experiment_times[-1]
    remaining_experiments = total_experiments - current_exp

    if remaining_experiments < 0:
        remaining_experiments = 0

    eta_seconds = last_exp_time * remaining_experiments

    # ETA
    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
    eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S")

    # remain
    remaining_str = format_time(eta_seconds)

    return f" | ETA: {eta_str} | Remaining: {remaining_str}"



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def log_args_table(args_dict, logger):
    # log hyperparameters
    table = Texttable()
    table.set_cols_align(["l", "l"])
    table.set_cols_dtype(['t', 't'])
    table.add_rows([["Argument", "Value"]] + list(args_dict.items()))
    logger.info("\nConfiguration:\n%s", table.draw())


logs = set()


def init_log(name, logpath, level=logging.INFO, session=None):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    format_str = "[%(asctime)s]  [%(levelname)s]  %(message)s"
    formatter = logging.Formatter(format_str)

    console_handler_exists = any(
        isinstance(h, logging.StreamHandler) and getattr(h, 'init_log_tag', None) == name
        for h in logger.handlers
    )
    if not console_handler_exists:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        console_handler.init_log_tag = name
        logger.addHandler(console_handler)

    timestamp = time.time()
    time_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(logpath + '/' + time_str + '_log_' + session + '.txt', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def fix_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True