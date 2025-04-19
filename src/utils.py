import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import os

import logging

logger = logging.getLogger()


def setup_logger(logfile, level=logging.INFO):
    """
    Sets up the logger to write to both the console and a log file.

    Args:
        logfile (str): Path to the log file.
        level (int): Logging level (default: logging.INFO).
    """
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the logging level
    logger.setLevel(level)

    # Create a file handler
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Set a logging format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def log_print(msg):
    print(msg)
    logger.info(msg)


class EmptyWith:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


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
        self.avg = self.sum / self.count


def cal_auc(pred, label):
    try:
        auc = roc_auc_score(label, pred)
        fprs, tprs, _ = roc_curve(label, pred)
        tdr = {}
        tdr["fpr"] = {}
        for t in [0.001, 0.0001]:
            ind = 0
            for fpr in fprs:
                if fpr > t:
                    break
                ind += 1
            tdr[t] = tprs[ind - 1]
            tdr["fpr"][t] = fprs[ind - 1]
    except Exception as e:
        log_print(e)
        auc = 0
        tdr = {0.001: 0, 0.0001: 0}
    return auc, tdr


def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def evaluate(output,label):
    output = softmax(output)

    pred_idx = np.argmax(output,axis=1)
    acc = np.sum(pred_idx==label)/len(label)
    auc, tdr = cal_auc(1-output[:,0],label)

    return acc, auc, tdr

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, s, s_hat):
        return torch.norm(s - s_hat, p=2)