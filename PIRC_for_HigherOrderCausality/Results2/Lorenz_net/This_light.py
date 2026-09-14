import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
import os
from Torch_Library import *
from Model.this import this,restrict_hypercube_size
from collections import defaultdict
def select_local_by_auto_delta(X, Y, keep_ratio=0.1, center=None, eps=1e-12):
    """
    Automatically choose delta so that at least keep_ratio samples
    are selected inside a hypercube.
    X: (T, N)
    Y: (T, N)
    keep_ratio: e.g. 0.1 means keep at least 10%.
    """
    if center is None:
        center = np.median(X, axis=0, keepdims=True)
    Z = X - center
    # Chebyshev distance to center:
    # inside hypercube iff max(abs(Z)) <= delta / 2
    r = np.max(np.abs(Z), axis=1)
    # choose threshold so that keep_ratio samples are retained
    threshold = np.quantile(r, keep_ratio)
    delta = 2 * (threshold + eps)
    inside = r <= delta / 2
    X_this = Z[inside]
    Y_this = Y[inside]
    ids = np.where(inside)[0]
    # print("=" * 50)
    # print(f"Auto delta       : {delta:.6g}")
    # print(f"Total samples    : {X.shape[0]}")
    # print(f"Selected samples : {len(ids)}")
    # print(f"Selected ratio   : {len(ids) / X.shape[0]:.4%}")
    # print("=" * 50)
    return X_this, Y_this, ids, delta, center

def objective_fast(trial, X=None, Y=None, args=None, T_Matrix=None, ooi=[], dmax=2):
    lam = trial.suggest_float("threshold", 1e-5, 1, log=True)
    rho = trial.suggest_float("alpha", 1e-5, 1, log=True)
    niter = trial.suggest_categorical("niter", [10])
    agg = trial.suggest_categorical("agg", ["max"])
    keep_ratio = trial.suggest_categorical("keep_ratio", [1.0])

    raw_node_num = int(args.node_num * 3)
    ExpandNodes = raw_node_num + math.comb(raw_node_num - 1, 2)

    X_this, Y_this, ids, delta, center = select_local_by_auto_delta(
        X,
        Y,
        keep_ratio=keep_ratio
    )

    Ainf, coeff, relerr = this(
        X_this.T, Y_this.T,  ooi, dmax,
        lam=lam, rho=rho, niter=10
    )
    T_from_Ainf = np.zeros((raw_node_num, ExpandNodes))

    com_dict = {}
    for i in range(raw_node_num):
        com_list = list(combinations([x for x in range(raw_node_num) if x != i], 2))
        com_dict[i] = {tuple(c): idx for idx, c in enumerate(com_list)}
    for order, mat in Ainf.items():
        if order == 2:
            for item in mat:
                i0, j0, index = int(item[0]), int(item[1]), item[2]
                if i0 != j0:
                    T_from_Ainf[i0, j0] = abs(index)
        elif order == 3:
            for item in mat:
                i0, j0, k0, index = int(item[0]), int(item[1]), int(item[2]), item[3]
                if i0 == j0 or i0 == k0 or j0 == k0:
                    continue
                pair = tuple(sorted((j0, k0)))
                idx = com_dict[i0][pair]
                T_from_Ainf[i0, raw_node_num + idx] = abs(index)

    Score = aggregate_causal_matrix(
        T_from_Ainf,
        group_size=3,
        agg=agg
    )

    AUC = ranking_auc_ALL(Score, T_Matrix)
    AUC = AUC.item() if hasattr(AUC, "item") else float(AUC)

    trial.set_user_attr("Score", np.asarray(Score))
    trial.set_user_attr("Ainf", Ainf)
    trial.set_user_attr("coeff", coeff)
    trial.set_user_attr("relerr", relerr)
    trial.set_user_attr("keep_ratio", keep_ratio)
    trial.set_user_attr("delta", delta)
    trial.set_user_attr("n_selected", len(ids))

    return AUC

def gridSearch_This(args=None,X=None,Y=None,T_Matrix=None):

    # ========= Optuna =========
    study = optuna.create_study(
        study_name=f"Lorenz_net",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=int(0.5 * args.n_trials)),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(lambda trial: objective_fast(trial,  X=X, Y=Y, args=args, T_Matrix=T_Matrix), n_trials=args.n_trials)

    best_trial = study.best_trial
    best_score = best_trial.user_attrs["Score"]

    print("\nBest Hyperparameters:", study.best_params)
    print("AUC:",best_trial.value)
    return study.best_params,best_score, best_trial.user_attrs
