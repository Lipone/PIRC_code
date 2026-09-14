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

def objective_fast(trial, X=None, Y=None, args=None, T_Matrix=None, base='polynomial'):
    # th = trial.suggest_float("th", 1e-7, 1, log=True)
    th=0.0001
    agg = trial.suggest_categorical("agg", ["max"])
    type = trial.suggest_categorical("type", ["mul","add"])
    basis_order = trial.suggest_categorical("basis_order", [2,3,4,5])

    raw_node_num = int(args.node_num * 3)
    ExpandNodes = raw_node_num + math.comb(raw_node_num - 1, 2)
    T_from_Ainf = np.zeros((raw_node_num, ExpandNodes))

    Y_T = Y.T

    expanded_cache = {}
    for NODE in range(raw_node_num):
        expanded_cache[(NODE, type)] = expand_nodes(X.copy(), NODE, type=type)

    results = Parallel(n_jobs=-1)(
        delayed(run_node)(
            NODE,
            type,
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

    trial.set_user_attr("Score", np.asarray(Score))

    return AUC


def gridSearch_ARNI(args=None, X=None, Y=None, T_Matrix=None, base='polynomial'):
    # ========= Optuna =========
    study = optuna.create_study(
        study_name=f"Lorenz_net",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=int(0.5 * args.n_trials)),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(lambda trial: objective_fast(trial, X=X, Y=Y, args=args, T_Matrix=T_Matrix, base=base),
                   n_trials=args.n_trials)

    best_trial = study.best_trial
    best_score = best_trial.user_attrs["Score"]

    print("\nBest Hyperparameters:", study.best_params)
    print("AUC:",best_trial.value)
    return study.best_params, best_score
