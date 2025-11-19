import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from joblib import Parallel, delayed
import os
import seaborn as sns
from joblib import Parallel, delayed
rng = np.random.default_rng(42)

def all_zero_combinations(n, k):
    rows = []
    for cols in itertools.combinations(range(n), k):
        row = np.zeros(n, dtype=int)
        row[list(cols)] = 1
        rows.append(row)
    return np.array(rows)

def K_order_input(x,k):
    K_order_input =  np.dot(all_zero_combinations(x.shape[-1], k), x)
    return K_order_input

def Partitioned_Win_Wres_1 (k, n):
    # k : number of inputs
    # n : number of reservoir nodes
    # k+math.comb(k-1, 2) : number of blocks
    block_num=k+math.comb(k-1, 2)
    Win_block_row=math.floor(n/block_num)
    Win_block_col=2
    #每个分块矩阵块的大小 row*col，每个分块矩阵元素满足-1至1均匀分布,生成block_num个分块矩阵，并拼接成一个对角矩阵
    Win_blocks = [rng.uniform(-1, 1, (Win_block_row, Win_block_col)) for _ in range(block_num)]
    Win = np.block([[Win_blocks[i] if i == j else np.zeros((Win_block_row, Win_block_col)) for j in range(block_num)] for i in range(block_num)])
    Wres_block_row= Wres_block_col= Win_block_row
    Wres_blocks = [rng.uniform(-1, 1, (Wres_block_row, Wres_block_col)) for _ in range(block_num)]
    Wres = np.block(
        [[Wres_blocks[i] if i == j else np.zeros((Wres_block_row, Wres_block_col)) for j in range(block_num)] for i in
         range(block_num)])
    return Win, Wres

def Structured_Matrix_W (input_dim):
    k=input_dim
    Structured_W_row= math.comb(k-1, 2)*2 + math.comb(k, 1)*2
    Structured_W_col= k+1
    mask = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1]
    ])
    Structured_W = rng.uniform(-1, 1, (Structured_W_row, Structured_W_col)) * mask
    return Structured_W

def generate_structured_w(cols):
    expand_dim = math.comb(cols-1, 2) * 2 + math.comb(cols, 1) * 2
    Structured_W = np.zeros((cols + 1, expand_dim), dtype=float)

    # Cause
    Structured_W[0, 0] = rng.uniform(-1, 1)
    Structured_W[1, 1] = rng.uniform(-1, 1)

    # Pairwise
    for p in range(1, cols):
        Structured_W[p+1, p*2] = rng.uniform(-1, 1)
        Structured_W[1, p*2+1] = rng.uniform(-1, 1)
        Structured_W[p+1, p*2+1] = rng.uniform(-1, 1)

    # Tri-order
    index_list = list(itertools.combinations(range(cols-1), 2))
    shifted_index_list = [(i + 2, j + 2) for i, j in index_list]
    for index, (q, w) in enumerate(shifted_index_list):
        Structured_W[1, (index+cols)*2+1] = rng.uniform(-1, 1)
        Structured_W[q, (index + cols) * 2] = rng.uniform(-1, 1)
        Structured_W[q, (index + cols) * 2 + 1] = rng.uniform(-1, 1)
        Structured_W[w, (index + cols) * 2] = rng.uniform(-1, 1)
        Structured_W[w, (index + cols) * 2 + 1] = rng.uniform(-1, 1)

    return Structured_W

def add_higher_order_terms(data, higher_order, is_1D=False):
    if higher_order:
        data = data.tolist()
        if is_1D:
            data=[data]
        for i in range(len(data)):
            for l in higher_order:
                higher_order_item = 1.0
                for index in l:
                    higher_order_item *= data[i][index]
                # 使用列表的 append() 方法
                data[i].append(higher_order_item)

        # 如果需要，可以转换回 numpy 数组
        data = np.array(data)
    return data

def shift_column_to_first(A, k):
    """
    将A的第k列移到第1列（索引0），原第1~(k-1)列依次后移
    :param A: 原始矩阵，shape (m, n)
    :param k: 需要移到第1列的列号（从0开始计数，且k>=1）
    :return: 新矩阵
    """
    # assert 1 <= k < A.shape[1], "k必须大于等于1且小于列数"

    col_k = A[:, k:k+1]      # 第k列
    left = A[:, :k]          # 第1到k-1列
    right = A[:, k+1:]       # k+1到最后一列

    # 拼接：第k列 + 原第1~(k-1)列 + 后面剩下的列
    new = np.hstack([col_k, left, right])
    return new




# 简单测试示例
if __name__ == "__main__":
    y_true = np.array([3.0, 5.0, 2.5, 7.0, 4.5, 6.0])
    y_pred = np.array([2.8, 5.2, 2.2, 6.5, 4.7, 6.1])

    for m in ["std", "range", "mean", "power"]:
        print(f"NRMSE({m}): {nrmse(y_true, y_pred, method=m):.6f}")











