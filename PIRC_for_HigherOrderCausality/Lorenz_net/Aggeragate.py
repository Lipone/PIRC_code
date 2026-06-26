import numpy as np
from itertools import combinations
from collections import defaultdict


def aggregate_causal_matrix(T_Matrix, group_size):
    """
    Aggregate causal matrix by merging every group_size nodes.

    Parameters
    ----------
    T_Matrix : np.ndarray
        Original causal matrix with shape:
        (node_num, node_num + C(node_num-1, 2))

        Columns:
        - first node_num columns: pairwise causality
        - remaining columns: triadic causality

    group_size : int
        Number of original nodes merged into one new node.
        Example: group_size=3 means
        0,1,2 -> 0
        3,4,5 -> 1
        6,7,8 -> 2

    Returns
    -------
    new_T_Matrix : np.ndarray
        Aggregated causal matrix.
    """

    node_num = T_Matrix.shape[0]

    assert node_num % group_size == 0, \
        "node_num must be divisible by group_size"

    new_node_num = node_num // group_size

    # =========================
    # 1. 原始矩阵 -> 字典
    # =========================
    causal_dict = defaultdict(list)

    for i in range(node_num):

        # pairwise
        for j in range(node_num):
            causal_dict[(i, j)].append(T_Matrix[i, j])

        # triadic
        com_list = list(combinations(
            [x for x in range(node_num) if x != i], 2
        ))

        for index, (j, k) in enumerate(com_list):
            value = T_Matrix[i, node_num + index]
            causal_dict[(i, j, k)].append(value)

    # =========================
    # 2. 聚合 key，相同 key 取最大值
    # =========================
    merged_dict = defaultdict(list)

    for key, values in causal_dict.items():
        new_key = tuple(v // group_size for v in key)
        merged_dict[new_key].extend(values)

    merged_dict = {
        k: int(max(v)) for k, v in merged_dict.items()
    }

    # =========================
    # 3. 字典 -> 新矩阵
    # =========================
    new_ExpandNodes = (
        new_node_num
        + (new_node_num - 1) * (new_node_num - 2) // 2
    )

    new_T_Matrix = np.zeros(
        (new_node_num, new_ExpandNodes),
        dtype=T_Matrix.dtype
    )

    tri_index = {}

    for i in range(new_node_num):
        com_list = list(combinations(
            [x for x in range(new_node_num) if x != i], 2
        ))
        tri_index[i] = {
            pair: idx for idx, pair in enumerate(com_list)
        }

    for key, value in merged_dict.items():

        # pairwise
        if len(key) == 2:
            target, source = key
            new_T_Matrix[target, source] = value

        # triadic
        elif len(key) == 3:
            target, j, k = key

            # 聚合后如果源节点包含 target，则跳过
            if j == target or k == target or j == k:
                continue

            j, k = sorted((j, k))

            idx = tri_index[target][(j, k)]
            new_T_Matrix[target, new_node_num + idx] = value

    return new_T_Matrix

import numpy as np
from itertools import combinations

# 原始节点数
node_num = 6

# 每3个聚合
group_size = 3

# 原始矩阵大小
ExpandNodes = node_num + (node_num - 1) * (node_num - 2) // 2

T = np.zeros((node_num, ExpandNodes), dtype=int)

# ------------------------
# pairwise
# ------------------------
# 0 -> 1
T[1, 0] = 1

# 2 -> 4
T[4, 2] = 1

# 3 -> 5
T[5, 3] = 1

# ------------------------
# triadic
# ------------------------
# 建立索引
tri_index = {}

for i in range(node_num):
    com_list = list(combinations(
        [x for x in range(node_num) if x != i], 2
    ))
    tri_index[i] = {c: idx for idx, c in enumerate(com_list)}

# {1,2} -> 0
idx = tri_index[0][(1,2)]
T[0, node_num + idx] = 1

# {3,5} -> 2
idx = tri_index[2][(3,5)]
T[2, node_num + idx] = 1

# {0,4} -> 5
idx = tri_index[5][(0,4)]
T[5, node_num + idx] = 1

print("Original T:")
print(T)

new_T = aggregate_causal_matrix(T, group_size=3)

print("\nAggregated T:")
print(new_T)