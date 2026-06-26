import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
from Model.PIRC import PIRC_flatten as Nonliear_PIRC
from DataGen import lorenz_nonpairwise
from scipy.integrate import solve_ivp
import numpy as np
from collections import defaultdict

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

import numpy as np
from itertools import combinations
from collections import defaultdict


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

def objective_fast(trial, data=None, args=None, T_Matrix=None):
    # ====== Hyperparameters ======
    n_units = trial.suggest_categorical('n_units', [5,10])
    alpha = trial.suggest_float('alpha', 0.1, 1, step=0.1)
    sigma_in = trial.suggest_float('sigma_in', 0.1, 1, step=0.1)
    rho = trial.suggest_float('rho', 0.1, 1, step=0.1)
    tikh = trial.suggest_categorical('tikh', [1.0,1e-1])
    bias = trial.suggest_categorical('bias', [0])
    option = trial.suggest_categorical('option', [2])
    tau = trial.suggest_categorical('tau', [2]) #在auc尺度下tau的选择几乎不影响结果，可以固定以加快搜索
    method = trial.suggest_categorical('method', ['Logistic']) #['exp', 'Logistic', 'power']
    connectivity = trial.suggest_categorical('connectivity', [1.0])
    I_type = trial.suggest_categorical('I_type', [1])
    mode = trial.suggest_categorical('mode', [2])
    block_dim = trial.suggest_categorical('block_dim', [1])
    agg = trial.suggest_categorical('agg', ['max', 'mean', "median"])

    device = torch.device("cuda:3")

    Win, Wres = Random_Win_Fixed_Wres(
        ExpandNodes+1, n_units, device,
        block_dim=block_dim, connectivity=connectivity
    )

    X = torch.as_tensor(data, dtype=torch.float32, device=device)
    X_batch = torch.stack(
        [shift_column_to_first(X, i) for i in range(args.node_num)],
        dim=0
    )
    X_washout, X_train, Y_train, Y_test = split_dataset2_batch(
        X_batch,
        args.N_washout,
        args.N_train,
        args.N_test,
        in_dim=args.node_num,
        out_dim=args.node_num
    )

    T_tensor = torch.as_tensor(T_Matrix, dtype=torch.float32, device=device)

    pirc = Nonliear_PIRC(
        n_units=n_units, in_dim=args.node_num, out_dim=args.node_num,
        Win=Win, Wres=Wres,
        sigma_in=sigma_in, rho=rho, alpha=alpha, tikh=tikh, mode=mode, bias=bias, dt=args.dt,
        device=device, block_dim=block_dim, I_type=I_type, option=option
    )

    R,_ = pirc.train(X_washout, X_train, Y_train)
    if mode == 1:
        Y_test_predict = pirc.Prediction(R, args.N_test, Y_train, Y_test)
    else:
        Y_test_predict = pirc.Prediction2(R, args.N_test, Y_train, Y_test)

    # ====== Score and AUC computation ======
    base_all = Y_test_predict[0, :, :args.N_test, 0]  # base: (B, T)
    others_all = Y_test_predict[2: , :, :args.N_test, 0]  # others: (E, B, T)
    base_expand = base_all.unsqueeze(0).expand_as(others_all)  # (E, B, T)
    score_EN = score_0_1_torch(
        others_all,
        base_expand,
        method=method,
        tau=tau
    )
    Score = score_EN.transpose(0, 1).contiguous()  # (B, E)
    Score = shift_column_for_Causal_Matrix(Score)
    Score = aggregate_causal_matrix(Score.detach().cpu().numpy(), group_size=3, agg=agg)
    Score = torch.as_tensor(Score, dtype=torch.float32, device=device)
    Score.fill_diagonal_(1.0)
    AUC = ranking_auc_ALL(Score, T_tensor)

    return AUC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_num', type=int, default=5, help='Number of nodes in the system')
    parser.add_argument("--N_train", type=int, default=1000)
    parser.add_argument("--N_test", type=int, default=10)
    parser.add_argument("--N_washout", type=int, default=0)
    parser.add_argument("--N_start", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--P_probability", type=float, default=0.5)
    parser.add_argument("--T_probability", type=float, default=0.5)
    parser.add_argument(
        "--Coupling_strength",
        type=float,
        default=0.1,
        help="Coupling strength for both pairwise and triadic interactions"
    )
    args = parser.parse_args()
    ExpandNodes = args.node_num + math.comb(args.node_num - 1, 2)
    set_seed(42)

    max_trials = 10
    for attempt in range(1, max_trials + 1):
        data, a2, a3 = gen_lorenz_net(args)
        result, _ = diagnose_lorenz_data(data, args.node_num)
        if not result["diverged"] and not result["synchronized"]:
            print(
                f"[Attempt {attempt}] Valid data generated. "
                f"max_abs={result['max_abs']:.3f}, "
                f"sync_error={result['sync_error_final']:.3e}"
            )
            break
        reason = []
        if result["diverged"]:
            reason.append("diverged")
        if result["synchronized"]:
            reason.append("synchronized")
        print(f"[Attempt {attempt}] Regenerate ({', '.join(reason)})")
    else:
        raise RuntimeError(
            f"Failed to generate a valid Lorenz dataset after {max_trials} attempts."
        )

    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    data = (data - mean) / std

    T_Matrix=np.zeros((args.node_num, ExpandNodes))
    T_Matrix[:, :args.node_num] = (a2 != 0).astype(int)
    for i in range(args.node_num):
        com_list=list(combinations([x for x in range(args.node_num) if x != i], 2))
        for index,(j,k) in enumerate(com_list):
            T_Matrix[i, args.node_num+index] = int(a3[i][j][k] != 0)
    print("ground_truth", T_Matrix)
    count_Pair = np.count_nonzero(T_Matrix[:, :args.node_num])
    count_Tri = np.count_nonzero(T_Matrix[:, args.node_num:])
    print("pair:", count_Pair)
    print("Tri:", count_Tri)

    args.node_num = args.node_num*3
    ExpandNodes = args.node_num + math.comb(args.node_num - 1, 2)

    # ========= Optuna =========
    study = optuna.create_study(
        study_name=f"Lorenz_net",
        direction="maximize",
        sampler = optuna.samplers.TPESampler(n_startup_trials=int(0.9*args.n_trials)),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(lambda trial: objective_fast(trial,data=data,args=args,T_Matrix=T_Matrix), n_trials=args.n_trials)

    print("Study finished.")
    print("\nBest Hyperparameters:", study.best_params)
    print("\nbest_AUC:", study.best_value)
    print("AUC level:", study.best_value)


