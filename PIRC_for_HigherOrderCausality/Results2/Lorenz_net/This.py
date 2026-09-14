import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
import os
from Torch_Library import *
from Model.this import this,restrict_hypercube_size
from collections import defaultdict

def objective_fast(trial, X=None, Y=None, args=None, T_Matrix=None, ooi=[], dmax=2):
    lam = trial.suggest_float("threshold", 1e-6, 1, log=True)
    rho = trial.suggest_float("alpha", 1e-6, 1, log=True)
    agg = trial.suggest_categorical("agg", ["max"])
    raw_node_num = int(args.node_num * 3)
    ExpandNodes = raw_node_num + math.comb(raw_node_num - 1, 2)

    Ainf, coeff, relerr = this(
        X.T, Y.T, ooi, dmax,
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

    return AUC

def gridSearch_This(args=None,X=None,Y=None,T_Matrix=None):

    # ========= Optuna =========
    study = optuna.create_study(
        study_name=f"Lorenz_net",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=int(0.9 * args.n_trials)),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(lambda trial: objective_fast(trial,  X=X, Y=Y, args=args, T_Matrix=T_Matrix), n_trials=args.n_trials)

    best_trial = study.best_trial
    best_score = best_trial.user_attrs["Score"]

    print("\nBest Hyperparameters:", study.best_params)
    print("AUC:",best_trial.value)
    return study.best_params,best_score
