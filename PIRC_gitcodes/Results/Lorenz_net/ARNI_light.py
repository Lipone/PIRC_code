import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
import os
from Torch_Library import *
from Model.ARNI import ARNI, expand_nodes
from sklearn.metrics import roc_curve, auc
from joblib import Parallel, delayed

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

def objective_fast(trial, X_full=None, Y_full=None, args=None, T_Matrix=None, base='polynomial'):
    agg = trial.suggest_categorical("agg", ["max"])
    th = trial.suggest_float("th", 1e-5, 1, log=True)
    typ = trial.suggest_categorical("type", ["mul", "add"])
    basis_order = trial.suggest_categorical("basis_order", [2,3,4,5])

    data_lengths = sorted(set([
         X_full.shape[0]
    ]))
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

    raw_node_num = int(args.node_num * 3)
    ExpandNodes = raw_node_num + math.comb(raw_node_num - 1, 2)

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

            T_from_Ainf = np.zeros((raw_node_num, ExpandNodes))
            Y_T = Y.T

            expanded_cache = {}
            for NODE in range(raw_node_num):
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
                for NODE in range(raw_node_num)
            )

            for NODE, vec in results:
                T_from_Ainf[NODE, :] = vec
            Score = aggregate_causal_matrix(
                T_from_Ainf,
                group_size=3,
                agg=agg
            )
            AUC = ranking_auc_ALL(Score, T_Matrix)
            AUC = AUC.item() if hasattr(AUC, "item") else float(AUC)

            seg_scores.append((AUC, Score.copy(), int(start), int(end)))
            seg_auc_records.append({
                "start": int(start),
                "end": int(end),
                "auc": float(AUC),
            })

            # print(
            #     f"[trial {trial.number}] "
            #     f"T_use={T_use}, start={start}, end={end}, AUC={AUC:.4f}"
            # )

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

        # print(
        #     f"[trial {trial.number}] "
        #     f"T_use={T_use}, length_best_auc={length_best_auc:.4f}, "
        #     f"best_window=({length_best_start}, {length_best_end})"
        # )

        if length_best_auc > best_auc + min_improve:
            best_auc = length_best_auc
            best_score = length_best_score.copy()
            best_T_use = T_use
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            # print(
            #     f"[trial {trial.number}] Early stop at T={T_use}, "
            #     f"global_best_auc={best_auc:.4f}, best_T_use={best_T_use}"
            # )
            break

    trial.set_user_attr("Score", np.asarray(best_score))
    trial.set_user_attr("auc_history", auc_history)
    trial.set_user_attr("best_T_use", best_T_use)

    return best_auc


def gridSearch_ARNI(args=None, X=None, Y=None, T_Matrix=None, base='polynomial'):
    # ========= Optuna =========
    study = optuna.create_study(
        study_name=f"Lorenz_net",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=int(0.5 * args.n_trials)),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(lambda trial: objective_fast(trial, X_full=X, Y_full=Y, args=args, T_Matrix=T_Matrix, base=base),
                   n_trials=args.n_trials)

    best_trial = study.best_trial
    best_score = best_trial.user_attrs["Score"]

    print("\nBest Hyperparameters:", study.best_params)
    print("AUC:",best_trial.value)
    return study.best_params, best_score, best_trial.user_attrs
