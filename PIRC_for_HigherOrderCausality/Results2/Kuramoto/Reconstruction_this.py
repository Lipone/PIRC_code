import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
import os
from Torch_Library import *
from DateGen_kuramoto import load_or_generate_kuramoto
from Model.this import this, restrict_hypercube_size

def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi
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
    print("=" * 50)
    print(f"Auto delta       : {delta:.6g}")
    print(f"Total samples    : {X.shape[0]}")
    print(f"Selected samples : {len(ids)}")
    print(f"Selected ratio   : {len(ids) / X.shape[0]:.4%}")
    print("=" * 50)
    return X_this, Y_this, ids, delta, center
def select_local_by_auto_delta_normalized(
    X, Y, keep_ratio=0.1, center=None, scale=None, eps=1e-8
):
    """
    EEG-style normalized local hypercube selection.

    X: (T, N)
    Y: (T, N)
    """

    if center is None:
        center = np.median(X, axis=0, keepdims=True)

    dX = X - center

    if scale is None:
        scale = np.median(np.abs(dX), axis=0, keepdims=True)
        scale = np.maximum(scale, eps)

    Z = dX / scale

    r = np.max(np.abs(Z), axis=1)

    threshold = np.quantile(r, keep_ratio)

    delta = 2 * (threshold + eps)

    inside = r <= delta / 2

    X_this = Z[inside]
    Y_this = Y[inside]
    ids = np.where(inside)[0]

    print("=" * 50)
    print("Normalized hypercube selection")
    print(f"Auto delta       : {delta:.6g}")
    print(f"Total samples    : {X.shape[0]}")
    print(f"Selected samples : {len(ids)}")
    print(f"Selected ratio   : {len(ids) / X.shape[0]:.4%}")
    print("=" * 50)

    return X_this, Y_this, ids, delta, center, scale
def objective_fast(trial, X=None, Y=None, args=None, T_Matrix=None, ooi=[], dmax=2, ExpandNodes=None):
    lam = trial.suggest_float("threshold", 1e-5, 1, log=True)
    rho = trial.suggest_float("alpha", 1e-5, 1, log=True)
    niter = trial.suggest_categorical("niter", [10])
    # keep_ratio = trial.suggest_categorical("keep_ratio", [1.0])

    # X_this, Y_this, ids, delta, center = select_local_by_auto_delta(
    #     X,
    #     Y,
    #     keep_ratio=keep_ratio
    # )
    # X_this=X_this% (2 * np.pi)
    Ainf, coeff, relerr = this(
        X.T, Y.T, ooi, dmax,
        lam=lam, rho=rho, niter=niter
    )
    T_from_Ainf = np.zeros((args.node_num, ExpandNodes))

    com_dict = {}
    for i in range(args.node_num):
        com_list = list(combinations([x for x in range(args.node_num) if x != i], 2))
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
                T_from_Ainf[i0, args.node_num + idx] = abs(index)

    AUC = ranking_auc_ALL(T_from_Ainf, T_Matrix)
    AUC = AUC.item() if hasattr(AUC, "item") else float(AUC)

    trial.set_user_attr("Score", np.asarray(T_from_Ainf))
    trial.set_user_attr("Ainf", Ainf)
    trial.set_user_attr("coeff", coeff)
    trial.set_user_attr("relerr", relerr)
    # trial.set_user_attr("keep_ratio", keep_ratio)
    # trial.set_user_attr("delta", delta)
    # trial.set_user_attr("n_selected", len(ids))

    return AUC

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
    parser.add_argument("--n_trials", type=int, default=200)
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

    X = X1
    Y = (X2 - X1) / args.dt

    # X1 = data[:-1, :]
    # X2 = data[1:, :]
    # # X = (X1 + X2) * 0.5
    X = X1 % (2 * np.pi)
    # Y = (data[1:, :] - data[:-1, :]) / args.dt

    study = optuna.create_study(
        study_name=f"kuramoto",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=int(0.5 * args.n_trials)),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(lambda trial: objective_fast(trial, X=X, Y=Y, args=args, T_Matrix=T_Matrix, ExpandNodes=ExpandNodes), n_trials=args.n_trials)

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

        # THIS 输出
        "Ainf": best_trial.user_attrs["Ainf"],
        "coeff": best_trial.user_attrs["coeff"],
        "relerr": best_trial.user_attrs["relerr"],

        # 局部超立方信息
        "keep_ratio": best_trial.user_attrs["keep_ratio"],
        "delta": best_trial.user_attrs["delta"],
        "n_selected": best_trial.user_attrs["n_selected"],
    }

    os.makedirs("results", exist_ok=True)

    with open("results/THIS_result_test2.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print("Results saved to results/THIS_result_test2.pkl")

