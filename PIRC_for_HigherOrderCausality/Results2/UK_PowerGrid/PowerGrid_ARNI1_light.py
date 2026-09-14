import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
import os
from Torch_Library import *
# from DateGen_kuramoto import load_or_generate_kuramoto
from Model.ARNI import ARNI, expand_nodes, basis_expansion
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
import pickle

def rrmse_per_node(Y_true, Y_pred, eps=1e-12):
    """
    Y_true, Y_pred: shape (N, T)
    return: node-wise RRMSE, shape (N,)
    """
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)

    rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=1))
    scale = np.sqrt(np.mean(Y_true ** 2, axis=1)) + eps

    return rmse / scale

def compute_arni_rrmse(
    X,
    Y,
    args,
    base="fourier",
    basis_order=5,
    expand_type="mul",
    th=1e-4
):
    """
    X: shape (T, N)
    Y: shape (T, N), derivative
    return:
        node-wise RRMSE, shape (N,)
        Y_pred, shape (N, T)
    """
    N = args.n
    Y_T = Y.T  # (N, T)

    Y_pred = np.zeros_like(Y_T)

    for NODE in range(N):
        X_expanded = expand_nodes(
            X.copy(),
            NODE,
            type=expand_type
        )

        X_expanded_T = X_expanded.T  # (ExpandedNodes, T)

        try:
            llist, cost, vec = ARNI(
                X_expanded_T,
                Y_T,
                base,
                basis_order,
                NODE,
                th=th
            )
        except np.linalg.LinAlgError:
            Y_pred[NODE, :] = 0.0
            continue

        X_basis = basis_expansion(
            X_expanded_T,
            basis_order,
            base,
            NODE
        )  # (num_basis, T, ExpandedNodes)

        if len(llist) == 0:
            Y_pred[NODE, :] = 0.0
            continue

        R = np.vstack([
            X_basis[:, :, idx]
            for idx in llist
        ])  # (num_selected_basis, T)

        try:
            RI = np.linalg.pinv(R)
            A = np.dot(Y_T[NODE, :], RI)
            Y_pred[NODE, :] = np.dot(A, R)
        except np.linalg.LinAlgError:
            Y_pred[NODE, :] = 0.0

    arni_rrmse = rrmse_per_node(Y_T, Y_pred)

    return arni_rrmse, Y_pred

def run_node(NODE, typ, base, order, X, Y_T, th=1e-4):
    try:
        X_expanded = expand_nodes(
            X.copy(),
            NODE,
            type=typ
        )

        llist, cost, vec = ARNI(
            X_expanded.T,
            Y_T,
            base,
            order,
            NODE,
            th=th
        )

        if not np.all(np.isfinite(vec)):
            vec = np.zeros(X_expanded.shape[1])

        return NODE, vec

    except np.linalg.LinAlgError:
        ExpandNodes = X.shape[1] + math.comb(X.shape[1] - 1, 2)
        return NODE, np.zeros(ExpandNodes)

    except Exception as e:
        ExpandNodes = X.shape[1] + math.comb(X.shape[1] - 1, 2)
        print(f"[WARN] NODE={NODE} failed: {e}")
        return NODE, np.zeros(ExpandNodes)

def read_data(args):
    the = pd.read_csv("./dataset/data/theta.csv").values[:, 1:].transpose()
    theta = np.zeros((args.N, args.n, args.T, 1))
    theta[0, :, :, 0] = the
    Sd = pd.read_csv("./dataset/data/Sd.csv").values[:, 1:]
    Xs = pd.read_csv("./dataset/data/trajectory.csv").values[:, 1:].transpose()
    Xss = np.zeros((args.n, args.N * args.T, args.V))
    Xsn = np.zeros((args.N, args.n, args.T, args.V))
    for i in range(args.V):
        Xss[:, :, i] = Xs[:, i * args.N * args.T:(i + 1) * args.N * args.T]
    for i in range(args.N):
        Xsn[i, :, :, :] = Xss[:, i * args.T:(i + 1) * args.T, :]
    time_point = pd.read_csv("./dataset/data/time_point.csv").values[:, 1:]
    edges_out = pd.read_csv("./dataset/data/edges.csv").values[:, 1:]
    Xsn = torch.tensor(Xsn).float()
    # Xsn = Xsn + args.ob_noise * np.random.randn(Xsn.shape[0], Xsn.shape[1], Xsn.shape[2], Xsn.shape[3])
    # X = Xsn[0, :, :, 0].numpy()
    # plt.plot(X[3].T,'-o')
    return (Xsn, theta, time_point, edges_out, Sd)

def objective_fast(
    trial,
    X_full=None,
    Y_full=None,
    args=None,
    T_Matrix=None,
    base='fourier',
    ExpandNodes=None,
):
    th = trial.suggest_float("th", 1e-5, 1e-4, log=True)
    typ = trial.suggest_categorical("type", ["mul","add"])
    basis_order = trial.suggest_categorical("basis_order", [5])

    data_lengths = [X_full.shape[0]]
    # data_lengths = [X_full.shape[0]]
    data_lengths = [L for L in data_lengths if L <= X_full.shape[0]]

    n_segments = 1
    min_improve = 1e-3
    patience = 0

    rng = np.random.default_rng(42 + trial.number)

    best_auc = -np.inf
    best_score = None
    best_T_use = None
    best_start = None
    best_end = None
    auc_history = []
    no_improve_count = 0

    for T_use in data_lengths:
        seg_scores = []
        seg_auc_records = []

        max_start = X_full.shape[0] - T_use

        if max_start <= 0:
            starts = [0]
        else:
            starts = rng.choice(
                np.arange(max_start + 1),
                size=min(n_segments, max_start + 1),
                replace=False
            )

        for start in starts:
            end = start + T_use

            X = X_full[start:end]
            Y = Y_full[start:end]

            T_from_Ainf = np.zeros((args.n, ExpandNodes))
            Y_T = Y.T

            # expanded_cache = {}
            # for NODE in range(args.n):
            #     expanded_cache[(NODE, typ)] = expand_nodes(
            #         X.copy(),
            #         NODE,
            #         type=typ
            #     )

            results = Parallel(n_jobs=-1)(
                delayed(run_node)(
                    NODE,
                    typ,
                    base,
                    basis_order,
                    X,
                    Y_T,
                    th=th
                )
                for NODE in range(args.n)
            )

            for NODE, vec in results:
                T_from_Ainf[NODE, :] = vec

            AUC = ranking_auc_ALL(T_from_Ainf, T_Matrix)
            AUC = AUC.item() if hasattr(AUC, "item") else float(AUC)

            seg_scores.append((AUC, T_from_Ainf.copy(), int(start), int(end)))
            seg_auc_records.append({
                "start": int(start),
                "end": int(end),
                "auc": float(AUC),
            })

            print(
                f"[trial {trial.number}] "
                f"T_use={T_use}, start={start}, end={end}, AUC={AUC:.4f}"
            )

        # 当前长度下取最大 AUC
        seg_scores.sort(key=lambda x: x[0], reverse=True)
        length_best_auc, length_best_score, length_best_start, length_best_end = seg_scores[0]

        auc_history.append({
            "T_use": int(T_use),
            "best_auc": float(length_best_auc),
            "best_start": int(length_best_start),
            "best_end": int(length_best_end),
            "segments": seg_auc_records,
        })

        print(
            f"[trial {trial.number}] "
            f"T_use={T_use}, length_best_auc={length_best_auc:.4f}, "
            f"best_window=({length_best_start}, {length_best_end})"
        )

        if length_best_auc > best_auc + min_improve:
            best_auc = length_best_auc
            best_score = length_best_score.copy()
            best_T_use = T_use
            best_start = length_best_start
            best_end = length_best_end
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(
                f"[trial {trial.number}] Early stop at T={T_use}, "
                f"global_best_auc={best_auc:.4f}, best_T_use={best_T_use}"
            )
            break

    trial.set_user_attr("Score", np.asarray(best_score))
    trial.set_user_attr("auc_history", auc_history)
    trial.set_user_attr("best_T_use", best_T_use)
    trial.set_user_attr("best_start", best_start)
    trial.set_user_attr("best_end", best_end)

    return best_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--data_ind', type=str, default='power_grid')
    parser.add_argument('--net_nam', type=str, default='edges')
    parser.add_argument('--direc', type=bool, default=True)
    # Parameters of experimental data
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--n', type=int, default=120)
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--V', type=int, default=2)
    parser.add_argument('--dt', type=float, default=0.08)
    parser.add_argument('--ddt', type=float, default=0.02)
    parser.add_argument("--N_train", type=int, default=100)
    parser.add_argument("--N_test", type=int, default=0)
    parser.add_argument("--N_washout", type=int, default=0)
    parser.add_argument("--N_start", type=int, default=100)
    parser.add_argument("--n_trials", type=int, default=3)
    args = parser.parse_args()
    (Xsn, theta, time_point, edges_out, Sd) = read_data(args)

    data=theta.squeeze().T #T*n
    ExpandNodes = args.n + math.comb(args.n - 1, 2)

    data = data[args.N_start:args.N_start + args.N_train,]

    T_Matrix = np.zeros((args.n, ExpandNodes))

    for [i, j] in edges_out:
        T_Matrix[int(i), int(j)] = 1
    for [i, j, k] in Sd:
        com_list = list(combinations([x for x in range(args.n) if x != int(i)], 2))
        idx = com_list.index(tuple(sorted((int(j), int(k)))))
        T_Matrix[int(i), args.n + idx] = 1
    print("ground_truth", T_Matrix)

    count_Pair = np.count_nonzero(T_Matrix[:, :args.n])
    count_Tri = np.count_nonzero(T_Matrix[:, args.n:])

    print("pair:", count_Pair)
    print("Tri:", count_Tri)

    X1 = data[:-1, :]
    X2 = data[1:, :]
    X = (X1 + X2) * 0.5
    X = X % (2 * np.pi)
    Y = (X2 - X1) / args.dt
    Y_T = Y.T

    study = optuna.create_study(
        study_name=f"GridPower_ARNI2",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=int(0.5 * args.n_trials)),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective_fast(
            trial,
            X_full=X,
            Y_full=Y,
            args=args,
            T_Matrix=T_Matrix,
            ExpandNodes=ExpandNodes
        ),
        n_trials=args.n_trials
    )
    best_trial = study.best_trial
    best_score = best_trial.user_attrs["Score"]
    best_params = study.best_params

    base = "fourier"
    typ = best_params["type"]
    basis_order = best_params["basis_order"]
    th = best_params["th"]
    start = best_trial.user_attrs["best_start"]
    end = best_trial.user_attrs["best_end"]

    # X_rrmse = X[start:end]
    # Y_rrmse = Y[start:end]

    # arni_rrmse, Y_pred = compute_arni_rrmse(
    #     X_rrmse,
    #     Y_rrmse,
    #     args,
    #     base=base,
    #     basis_order=basis_order,
    #     expand_type=typ,
    #     th=th
    # )

    # print("Best AUC:", study.best_value)
    # print("Mean node-wise RRMSE:", np.mean(arni_rrmse))

    save_dict = {
        "args": vars(args),
        "best_params": best_params,
        "best_auc": study.best_value,

        "Score": best_score,
        "T_Matrix": T_Matrix,

        # "RRMSE": arni_rrmse,
        # "Mean_RRMSE": float(np.mean(arni_rrmse)),
        # "Y_pred": Y_pred,
        # "Y_true": Y_rrmse.T,
        #
        "best_T_use": best_trial.user_attrs.get("best_T_use", None),
        "best_start": start,
        "best_end": end,
        # "X_rrmse": X_rrmse,
        # "auc_history": best_trial.user_attrs.get("auc_history", None),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/ARNI2_result_test.pkl", "wb") as f:
        pickle.dump(save_dict, f)
    print("Results saved to results/ARNI2_result_test.pkl")