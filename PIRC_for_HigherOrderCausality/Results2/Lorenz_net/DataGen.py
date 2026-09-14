import numpy as np
from scipy.integrate import solve_ivp
from collections import defaultdict
from itertools import combinations


def lorenz_nonpairwise(t, state, A2, A3, sigma=10.0, rho=28.0, beta=8/3):
    """
    Lorenz oscillators with pairwise and triadic interactions.

    state: shape (3*n,)
           [x1,...,xn, y1,...,yn, z1,...,zn]
    A2: shape (n,n)
        pairwise adjacency matrix a_ij^(2)
    A3: shape (n,n,n)
        triadic adjacency tensor a_ijk^(3)
    """

    n = A2.shape[0]

    x = state[:n]
    y = state[n:2*n]
    z = state[2*n:3*n]

    # intrinsic Lorenz dynamics
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    # pairwise coupling: sum_j a_ij^(2) (x_j - x_i)
    pairwise = A2 @ x - x * A2.sum(axis=1)

    # triadic coupling: sum_{j,k} a_ijk^(3) (x_j x_k^2 - x_i^3)
    triadic_drive = np.einsum("ijk,j,k->i", A3, x, x**2)
    triadic_self = x**3 * A3.sum(axis=(1, 2))
    triadic = triadic_drive - triadic_self

    dx += pairwise + triadic

    return np.concatenate([dx, dy, dz])


# # =========================
# # Example usage
# # =========================
#
# n = 10
# T = 50
# dt = 0.01
#
# # random pairwise adjacency
# A2 = np.random.rand(n, n) < 0.2
# A2 = A2.astype(float)
# np.fill_diagonal(A2, 0.0)
#
# # random triadic adjacency tensor
# A3 = np.random.rand(n, n, n) < 0.02
# A3 = A3.astype(float)
#
# # remove self-related triadic terms if desired
# for i in range(n):
#     A3[i, i, :] = 0.0
#     A3[i, :, i] = 0.0
#
# # coupling strengths
# gamma2 = 0.1
# gamma3 = 0.01
# A2 *= gamma2
# A3 *= gamma3
#
# # initial condition
# state0 = np.random.randn(3 * n)
#
# t_eval = np.arange(0, T, dt)
#
# sol = solve_ivp(
#     lorenz_nonpairwise,
#     t_span=(0, T),
#     y0=state0,
#     t_eval=t_eval,
#     args=(A2, A3),
#     method="RK45",
#     rtol=1e-8,
#     atol=1e-10
# )
#
# X = sol.y[:n].T
# Y = sol.y[n:2*n].T
# Z = sol.y[2*n:3*n].T
#
# print(X.shape)  # (time_steps, n)
def diagnose_lorenz_data(data, n, div_threshold=1e4, sync_threshold=1e-2):
    result = {}

    result["finite"] = np.all(np.isfinite(data))
    result["max_abs"] = np.max(np.abs(data))

    result["diverged"] = (
        (not result["finite"]) or
        (result["max_abs"] > div_threshold)
    )

    X = data.reshape(data.shape[0], n, 3)

    mean_state = X.mean(axis=1, keepdims=True)
    sync_err = np.linalg.norm(X - mean_state, axis=2).mean(axis=1)

    result["sync_error_mean_last_half"] = sync_err[len(sync_err)//2:].mean()
    result["sync_error_final"] = sync_err[-1]

    result["synchronized"] = (
        result["sync_error_mean_last_half"] < sync_threshold
    )

    return result, sync_err

def diagnose_time_points(data, raw_n, div_threshold=1e4, sync_threshold=1e-2):
    # data: (T, 3*raw_n)
    X = data.reshape(data.shape[0], raw_n, 3)

    # 每个时刻的最大幅值，用来看发散
    amp = np.max(np.abs(data), axis=1)

    # 每个时刻的同步误差
    mean_state = X.mean(axis=1, keepdims=True)
    sync_err = np.linalg.norm(X - mean_state, axis=2).mean(axis=1)

    # 第一次发散时刻
    div_idx = np.where((~np.isfinite(amp)) | (amp > div_threshold))[0]
    div_start = div_idx[0] if len(div_idx) > 0 else None

    # 第一次同步时刻
    sync_idx = np.where(sync_err < sync_threshold)[0]
    sync_start = sync_idx[0] if len(sync_idx) > 0 else None

    return {
        "div_start": div_start,
        "sync_start": sync_start,
        "max_amp": np.nanmax(amp),
        "final_sync_error": sync_err[-1],
    }, amp, sync_err

def gen_lorenz_net(args):
    n = args.node_num
    dt = args.dt
    T = (args.N_train + args.N_test + args.N_washout + args.N_start + 1) * args.dt

    A2 = (np.random.rand(n, n) < args.P_probability).astype(float)

    np.fill_diagonal(A2, 0.0)

    A2 *= args.Coupling_strength

    # triadic coupling

    mask = np.random.rand(n, n, n) < args.T_probability

    A3 = np.zeros((n, n, n))

    upper = np.triu(np.ones((n, n)), k=1).astype(bool)

    upper3 = upper[None, :, :]

    A3[mask & upper3] = args.Coupling_strength

    for i in range(n):
        A3[i, i, :] = 0.0
        A3[i, :, i] = 0.0

    state0 = np.random.randn(3 * n)
    t_eval = np.arange(0, T, dt)

    sol = solve_ivp(
        lorenz_nonpairwise,
        t_span=(0, T),
        y0=state0,
        t_eval=t_eval,
        args=(A2, A3),
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )

    Y = sol.y.T  # (T, 3n), [x1...xn, y1...yn, z1...zn]

    x = Y[:, :n]
    y = Y[:, n:2*n]
    z = Y[:, 2*n:3*n]

    data = np.stack([x, y, z], axis=2).reshape(Y.shape[0], 3*n)

    return data, A2, A3

def aggregate_causal_matrix(T_Matrix, group_size, agg="max"):

    node_num = T_Matrix.shape[0]
    assert node_num % group_size == 0

    new_node_num = node_num // group_size

    def canonical_key(key):
        key = [v // group_size for v in key]

        target = key[0]
        sources = key[1:]

        # 去掉和 target 相同的 source
        sources = [s for s in sources if s != target]

        # source 去重
        sources = sorted(set(sources))

        if len(sources) == 0:
            return None

        return tuple([target] + sources)

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
            causal_dict[(i, j, k)].append(
                T_Matrix[i, node_num + index]
            )

    merged_dict = defaultdict(list)

    for key, values in causal_dict.items():
        new_key = canonical_key(key)
        if new_key is None:
            continue
        merged_dict[new_key].extend(values)

    if agg == "max":
        merged_dict = {k: np.max(v) for k, v in merged_dict.items()}
    elif agg == "mean":
        merged_dict = {k: np.mean(v) for k, v in merged_dict.items()}
    elif agg == "median":
        merged_dict = {k: np.median(v) for k, v in merged_dict.items()}

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

        if len(key) == 2:
            target, source = key
            if target != source:
                new_T_Matrix[target, source] = value

        elif len(key) == 3:
            target, j, k = key
            j, k = sorted((j, k))

            if j != target and k != target and j != k:
                idx = tri_index[target][(j, k)]
                new_T_Matrix[target, new_node_num + idx] = value

    return new_T_Matrix