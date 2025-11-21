import itertools
import math
from sklearn.metrics import mean_squared_error
import argparse

import optuna
import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Lorenz.Data_gen import *
import pickle
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from datetime import date
from pathlib import Path
from filelock import FileLock
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue

import random
import numpy as np
import torch
import torch.multiprocessing as mp
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def Partitioned_Win_Wres(n, ExpandNodes, device, block_dim = 1):  # n : number of reservoir nodes
    block_num = ExpandNodes
    Win_block_row = math.floor(n / block_num)
    Win_block_col = 2 * block_dim
    Win_blocks = [torch.empty(Win_block_row, Win_block_col, device=device).uniform_(-1, 1) for _ in
                  range(block_num)]
    Win = torch.block_diag(*Win_blocks)
    Wres_block_row = Wres_block_col = Win_block_row
    Wres_blocks = [torch.empty(Wres_block_row, Wres_block_col, device=device).uniform_(-1, 1) for _ in
                   range(block_num)]
    Wres = torch.block_diag(*Wres_blocks)
    return Win, Wres

def Partitioned_Win_Wres_sameBlocks(n, ExpandNodes, device, block_dim = 1):
    block_num = ExpandNodes
    Win_block_row = math.floor(n / block_num)
    Win_block_col = 2 * block_dim
    # 每个分块矩阵块的大小 row*col，每个分块矩阵元素满足-1至1均匀分布,生成block_num个分块矩阵，并拼接成一个对角矩阵
    Win_block = torch.empty(Win_block_row, Win_block_col, device=device).uniform_(-1, 1)
    Win_blocks = [Win_block.clone() for _ in range(block_num)]
    Win = torch.block_diag(*Win_blocks)
    Wres_block_row = Wres_block_col = Win_block_row
    Wres_block = torch.empty(Wres_block_row, Wres_block_col, device=device).uniform_(-1, 1)
    Wres_blocks = [Wres_block.clone() for _ in range(block_num)]
    Wres = torch.block_diag(*Wres_blocks)
    return Win, Wres

def generate_structured_w_v1(cols, ExpandNodes, device): # cols : number of inputs  #original version (all random)
    ExpandDim = ExpandNodes * 2
    Structured_W = torch.zeros((cols + 1, ExpandDim), device=device)
    Structured_W[0, 0] = torch.empty(1).uniform_(-1, 1)
    Structured_W[1, 1] = torch.empty(1).uniform_(-1, 1)
    for p in range(1, cols):
        Structured_W[p + 1, p * 2] = torch.empty(1).uniform_(-1, 1)
        Structured_W[1, p * 2 + 1] = torch.empty(1).uniform_(-1, 1)
        Structured_W[p + 1, p * 2 + 1] = torch.empty(1).uniform_(-1, 1)
    index_list = list(itertools.combinations(range(cols - 1), 2))
    shifted_index_list = [(i + 2, j + 2) for i, j in index_list]
    for index, (q, w) in enumerate(shifted_index_list):
        Structured_W[1, (index + cols) * 2 + 1] = torch.empty(1).uniform_(-1, 1)
        Structured_W[q, (index + cols) * 2] = torch.empty(1).uniform_(-1, 1)
        Structured_W[q, (index + cols) * 2 + 1] = torch.empty(1).uniform_(-1, 1)
        Structured_W[w, (index + cols) * 2] = torch.empty(1).uniform_(-1, 1)
        Structured_W[w, (index + cols) * 2 + 1] = torch.empty(1).uniform_(-1, 1)
    return Structured_W

def set_block(mat, row, col, block, block_dim = 1):
    mat[row * block_dim:row * block_dim + block_dim, col * block_dim:col * block_dim + block_dim] = block

def generate_structured_w_v2(cols, ExpandNodes, device, block_dim = 1): # cols : number of inputs #extend version (all random)
    Structured_W = torch.zeros(((cols + 1) * block_dim, ExpandNodes * 2 * block_dim), device=device)
    set_block(Structured_W, 0, 0, torch.empty(block_dim, block_dim, device=device).uniform_(-1, 1), block_dim)
    set_block(Structured_W, 1, 1, torch.empty(block_dim, block_dim, device=device).uniform_(-1, 1), block_dim)
    for p in range(1, cols):
        set_block(Structured_W, p + 1, p * 2, torch.empty(block_dim, block_dim, device=device).uniform_(-1, 1), block_dim)
        set_block(Structured_W, 1, p * 2 + 1, torch.empty(block_dim, block_dim, device=device).uniform_(-1, 1), block_dim)
        set_block(Structured_W, p + 1, p * 2 + 1, torch.empty(block_dim, block_dim, device=device).uniform_(-1, 1), block_dim)
    index_list = list(itertools.combinations(range(cols - 1), 2))
    shifted_index_list = [(i + 2, j + 2) for i, j in index_list]
    for index, (q, w) in enumerate(shifted_index_list):
        set_block(Structured_W, 1, (index + cols) * 2 + 1, torch.empty(block_dim, block_dim, device=device).uniform_(-1, 1), block_dim)
        set_block(Structured_W, q, (index + cols) * 2, torch.empty(block_dim, block_dim, device=device).uniform_(-1, 1), block_dim)
        set_block(Structured_W, q, (index + cols) * 2 + 1, torch.empty(block_dim, block_dim, device=device).uniform_(-1, 1), block_dim)
        set_block(Structured_W, w, (index + cols) * 2, torch.empty(block_dim, block_dim, device=device).uniform_(-1, 1), block_dim)
        set_block(Structured_W, w, (index + cols) * 2 + 1, torch.empty(block_dim, block_dim, device=device).uniform_(-1, 1), block_dim)
    return Structured_W

def generate_structured_w_v3(cols, ExpandNodes, device, block_dim = 1): # cols : number of inputs #extend version (all ones)
    Structured_W = torch.zeros(((cols + 1) * block_dim, ExpandNodes * 2 * block_dim), device=device)
    set_block(Structured_W, 0, 0, torch.ones(block_dim, block_dim, device=device), block_dim)
    set_block(Structured_W, 1, 1, torch.ones(block_dim, block_dim, device=device), block_dim)
    for p in range(1, cols):
        set_block(Structured_W, p + 1, p * 2, torch.ones(block_dim, block_dim, device=device), block_dim)
        set_block(Structured_W, 1, p * 2 + 1, torch.ones(block_dim, block_dim, device=device), block_dim)
        set_block(Structured_W, p + 1, p * 2 + 1, torch.ones(block_dim, block_dim, device=device), block_dim)
    index_list = list(itertools.combinations(range(cols - 1), 2))
    shifted_index_list = [(i + 2, j + 2) for i, j in index_list]
    for index, (q, w) in enumerate(shifted_index_list):
        set_block(Structured_W, 1, (index + cols) * 2 + 1, torch.ones(block_dim, block_dim, device=device), block_dim)
        set_block(Structured_W, q, (index + cols) * 2, torch.ones(block_dim, block_dim, device=device), block_dim)
        set_block(Structured_W, q, (index + cols) * 2 + 1, torch.ones(block_dim, block_dim, device=device), block_dim)
        set_block(Structured_W, w, (index + cols) * 2, torch.ones(block_dim, block_dim, device=device), block_dim)
        set_block(Structured_W, w, (index + cols) * 2 + 1, torch.ones(block_dim, block_dim, device=device), block_dim)
    return Structured_W

def generate_structured_w_v4(cols, ExpandNodes, device, block_dim = 1): # cols : number of inputs #extend version (all diag)
    Structured_W = torch.zeros(((cols + 1) * block_dim, ExpandNodes * 2 * block_dim), device=device)
    diag_block = torch.eye(block_dim, device=device)
    set_block(Structured_W, 0, 0, diag_block, block_dim)
    set_block(Structured_W, 1, 1, diag_block, block_dim)
    for p in range(1, cols):
        set_block(Structured_W, p + 1, p * 2, diag_block, block_dim)
        set_block(Structured_W, 1, p * 2 + 1, diag_block, block_dim)
        set_block(Structured_W, p + 1, p * 2 + 1, diag_block, block_dim)
    index_list = list(itertools.combinations(range(cols - 1), 2))
    shifted_index_list = [(i + 2, j + 2) for i, j in index_list]
    for index, (q, w) in enumerate(shifted_index_list):
        set_block(Structured_W, 1, (index + cols) * 2 + 1, diag_block, block_dim)
        set_block(Structured_W, q, (index + cols) * 2, diag_block, block_dim)
        set_block(Structured_W, q, (index + cols) * 2 + 1, diag_block, block_dim)
        set_block(Structured_W, w, (index + cols) * 2, diag_block, block_dim)
        set_block(Structured_W, w, (index + cols) * 2 + 1, diag_block, block_dim)
    return Structured_W

#split dataset
def split_dataset(X, N_start, N_washout, N_train, N_test, in_dim=3, out_dim=3):
    X = X[N_start:, :]
    X_washout = X[:N_washout, : in_dim]
    X_train = X[N_washout:N_washout + N_train, : in_dim]
    Y_train = X[N_washout + 1:N_washout + N_train + 1, : out_dim]
    Y_test = X[N_washout + N_train + 1: N_washout + N_train + N_test + 1, : out_dim]
    return X_washout, X_train, Y_train, Y_test

def shift_column_to_first(A, k):
    col_k = A[:, k:k+1]      # 第k列
    left = A[:, :k]          # 第1到k-1列
    right = A[:, k+1:]       # k+1到最后一列
    # 拼接：第k列 + 原第1~(k-1)列 + 后面剩下的列
    new = torch.cat([col_k, left, right], dim=1)
    return new

def shift_column_for_Causal_Matrix(A: torch.Tensor) -> torch.Tensor:
    is_numpy = isinstance(A, np.ndarray)
    n, m = A.shape
    base = A.copy() if is_numpy else A.clone()
    for i in range(n):
        diag_val = base[i, 0]
        left = base[i, 1:i+1]
        right = base[i, i+1:]
        if is_numpy:
            new_row = np.concatenate((left, [diag_val], right), axis=0)
            A[i, :] = new_row
        else:
            new_row = torch.cat([left, torch.tensor([diag_val], device=base.device, dtype=base.dtype),
                                 right], dim=0)
            A[i, :] = new_row
    return A

def nmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true)
    return min(mse / (var + 1e-10), 1.0)

def sin_cos_theta(theta_vector):
    new_vector = torch.zeros(theta_vector.size(0), theta_vector.size(1) * 2, device=theta_vector.device)
    new_vector[:, 0::2] = torch.sin(theta_vector)
    new_vector[:, 1::2] = torch.cos(theta_vector)
    return new_vector

def sin_theta(theta_vector):
    new_vector = torch.sin(theta_vector)
    return new_vector

def stop_when_low_enough(study, trial, THRESHOLD=1e-4):
    # 在每个 trial 之后调用；若已知 best_value 且小于阈值则停止所有优化
    try:
        best = study.best_value
    except Exception:
        best = None
    if best is None:
        return
    # 保护性检查，避免 NaN 比较
    if isinstance(best, float) and not math.isnan(best) and best < THRESHOLD:
        print(f"Early stopping: best_value={best:.3e} < {THRESHOLD:.3e} -> stopping study.")
        study.stop()

def _is_torch(x):
    return isinstance(x, torch.Tensor)

def _to_float(x):
    if _is_torch(x):
        return x if x.is_floating_point() else x.float()
    return x if np.issubdtype(x.dtype, np.floating) else x.astype(np.float32)

def score_from_mse_large_better(y_pred, y_true, method="exp", eps=1e-12):
    """
    将 MSE 映射到 [0,1]，且 MSE 越大得分越高。
    method: 'exp' | 'ratio' | 'rel_worse'
    - exp:        1 - exp(-MSE/scale)
    - ratio:      MSE / (MSE + scale)
    - rel_worse:  clip(MSE / baseline_mse, 0, 1)
    """
    y_pred = _to_float(y_pred)
    y_true = _to_float(y_true)

    if _is_torch(y_true):
        mse = ((y_pred - y_true) ** 2).mean()
        var = y_true.var(unbiased=False)
        energy = (y_true ** 2).mean()
        scale = torch.clamp(var, min=eps)
        if scale <= eps:
            scale = torch.clamp(energy, min=1.0)

        if method == "exp":
            score = 1.0 - torch.exp(-mse / scale)
        elif method == "ratio":
            score = mse / (mse + scale)
        elif method == "rel_worse":
            mu = y_true.mean(dim=0, keepdim=True)
            baseline_mse = ((mu - y_true) ** 2).mean().clamp_min(eps)
            score = (mse / baseline_mse).clamp(0.0, 1.0)
        else:
            raise ValueError("unknown method")
        return score
    else:
        mse = float(np.mean((y_pred - y_true) ** 2))
        var = float(np.var(y_true))
        energy = float(np.mean(y_true ** 2))
        scale = max(var, eps)
        if scale <= eps:
            scale = max(energy, 1.0)

        if method == "exp":
            score = 1.0 - np.exp(-mse / scale)
        elif method == "ratio":
            score = mse / (mse + scale)
        elif method == "rel_worse":
            mu = np.mean(y_true, axis=0, keepdims=True)
            baseline_mse = max(float(np.mean((mu - y_true) ** 2)), eps)
            score = np.clip(mse / baseline_mse, 0.0, 1.0)
        else:
            raise ValueError("unknown method")
        return float(score)


