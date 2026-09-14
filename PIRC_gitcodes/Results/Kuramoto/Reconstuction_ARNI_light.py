import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
import os
from Torch_Library import *
from Model.ARNI import ARNI, expand_nodes
from joblib import Parallel, delayed
from DateGen_kuramoto import load_or_generate_kuramoto

def run_node(NODE, typ, base, order, expanded_cache, Y_T, th=1e-4):
    X_expanded = expanded_cache[(NODE, typ)]

    llist, cost, vec = ARNI(
        X_expanded.T,
        Y_T,
        base,
        order,
        NODE,
        th=th
    )

    return NODE, vec

def objective_fast(
    trial,
    X_full=None,
    Y_full=None,
    args=None,
    T_Matrix=None,
    base='fourier',
    ExpandNodes=None,
):
    th = trial.suggest_float("th", 1e-5, 1, log=True)
    typ = trial.suggest_categorical("type", ["mul", "add"])
    basis_order = trial.suggest_categorical("basis_order", [2,3,4,5])

    data_lengths = [5,10,15,20,25,30,35,40,50,100,150,200,300,500,X_full.shape[0]]
    data_lengths = [L for L in data_lengths if L <= X_full.shape[0]]

    n_segments = 5
    min_improve = 1e-3
    patience = 3

    rng = np.random.default_rng(42 + trial.number)

    best_auc = -np.inf
    best_score = None
    best_T_use = None
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

            T_from_Ainf = np.zeros((args.node_num, ExpandNodes))
            Y_T = Y.T

            expanded_cache = {}
            for NODE in range(args.node_num):
                expanded_cache[(NODE, typ)] = expand_nodes(
                    X.copy(),
                    NODE,
                    type=typ
                )

            results = Parallel(n_jobs=-1)(
                delayed(run_node)(
                    NODE,
                    typ,
                    base,
                    basis_order,
                    expanded_cache,
                    Y_T,
                    th=th
                )
                for NODE in range(args.node_num)
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

    return best_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_num', type=int, default=20, help='Number of nodes in the system')
    parser.add_argument('--Pair_strength', type=float, default=0.4, help='Pairwise interaction strength') #1
    parser.add_argument('--Tri_strength', type=float, default=0.4, help='Three-way interaction strength') #1
    parser.add_argument("--N_train", type=int, default=1000)
    parser.add_argument("--N_test", type=int, default=0)
    parser.add_argument("--N_washout", type=int, default=0)
    parser.add_argument("--N_start", type=int, default=0)
    parser.add_argument("--P_probability", type=float, default=0.05) #0.02
    parser.add_argument("--T_probability", type=float, default=0.005)  # 0.02
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()
    set_seed(42)

    ExpandNodes = args.node_num + math.comb(args.node_num - 1, 2)
    ooi = []
    dmax = 2

    data, a2, a3 = load_or_generate_kuramoto(args)

    data = data[args.N_start:,]

    T_Matrix = np.zeros((args.node_num, ExpandNodes))
    T_Matrix[:, :args.node_num] = (a2 != 0).astype(int)
    for i in range(args.node_num):
        com_list = list(combinations([x for x in range(args.node_num) if x != i], 2))
        for index, (j, k) in enumerate(com_list):
            T_Matrix[i, args.node_num + index] = int(a3[i][j][k] != 0)
    print("ground_truth", T_Matrix)

    count_Pair = np.count_nonzero(T_Matrix[:, :args.node_num])

    # 统计剩下列的非零元素
    count_Tri = np.count_nonzero(T_Matrix[:, args.node_num:])

    print("pair:", count_Pair)
    print("Tri:", count_Tri)

    X1 = data[:-1, :]
    X2 = data[1:, :]
    X = (X1 + X2) * 0.5
    X = X % (2 * np.pi)
    Y = (data[1:, :] - data[:-1, :]) / args.dt

    study = optuna.create_study(
        study_name=f"Kuramoto_ARNI",
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
    import pickle
    import os

    save_dict = {
        "args": vars(args),
        "best_params": study.best_params,
        "best_auc": study.best_value,
        "Score": best_score,
        "T_Matrix": T_Matrix,
        "a2": a2,
        "a3": a3,
        "best_T_use": best_trial.user_attrs.get("best_T_use", None),
        "auc_history": best_trial.user_attrs.get("auc_history", None),
    }

    os.makedirs("results", exist_ok=True)

    with open("results/ARNI2_light_result.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print("Results saved to results/ARNI2_light_result.pkl")
