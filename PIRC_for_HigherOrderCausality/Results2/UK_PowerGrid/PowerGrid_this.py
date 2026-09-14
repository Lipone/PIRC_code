import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
import os
from Torch_Library import *
from Model.this import this, get_thetad
import pandas as pd

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
def compute_this_rrmse(
    X_this,
    Y_this,
    dmax=2,
    lam=0.1,
    rho=1.0,
    niter=10
):
    """
    X_this: shape (N, T)
    Y_this: shape (N, T), derivative

    return:
        node-wise RRMSE, shape (N,)
        Y_pred, shape (N, T)
        Ainf, coeff
    """
    Ainf, coeff, relerr = this(
        X_this,
        Y_this,
        ooi=[],
        dmax=dmax,
        lam=lam,
        rho=rho,
        niter=niter
    )

    theta, _ = get_thetad(X_this, dmax)
    Y_pred = coeff @ theta

    this_rrmse = rrmse_per_node(Y_this, Y_pred)

    return this_rrmse, Y_pred, Ainf, coeff

def objective_fast(trial, X=None, Y=None, args=None, T_Matrix=None, ooi=[], dmax=2, ExpandNodes=None):
    lam = trial.suggest_float("threshold", 1e-5, 1, log=True)
    rho = trial.suggest_float("alpha", 1e-5, 1, log=True)
    niter = trial.suggest_categorical("niter", [10])

    Ainf, coeff, relerr = this(
        X.T, Y.T, ooi, dmax,
        lam=lam, rho=rho, niter=niter
    )
    T_from_Ainf = np.zeros((args.n, ExpandNodes))

    com_dict = {}
    for i in range(args.n):
        com_list = list(combinations([x for x in range(args.n) if x != i], 2))
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
                T_from_Ainf[i0, args.n + idx] = abs(index)

    AUC = ranking_auc_ALL(T_from_Ainf, T_Matrix)
    AUC = AUC.item() if hasattr(AUC, "item") else float(AUC)

    trial.set_user_attr("Score", np.asarray(T_from_Ainf))
    trial.set_user_attr("Ainf", Ainf)
    trial.set_user_attr("coeff", coeff)
    trial.set_user_attr("relerr", relerr)

    return AUC

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
    parser.add_argument("--n_trials", type=int, default=10)
    args = parser.parse_args()
    (Xsn, theta, time_point, edges_out, Sd) = read_data(args)

    data=theta.squeeze().T #T*n
    ExpandNodes = args.n + math.comb(args.n - 1, 2)

    ooi = []
    dmax = 2

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

    # X = data[:-1, :]
    # X = X % (2 * np.pi)
    # Y = (data[1:, :] - data[:-1, :]) / args.dt

    X1 = data[:-1, :]
    X2 = data[1:, :]
    X = X1
    X = (X+np.pi) % (2 * np.pi) -np.pi
    Y = (X2 - X1) / args.dt

    study = optuna.create_study(
        study_name=f"PowerGrid",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=int(0.5 * args.n_trials)),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(lambda trial: objective_fast(trial, X=X, Y=Y, args=args, T_Matrix=T_Matrix, ExpandNodes=ExpandNodes),
                   n_trials=args.n_trials)

    best_trial = study.best_trial
    best_score = best_trial.user_attrs["Score"]
    best_params = study.best_params

    # best_ids = best_trial.user_attrs["ids"]
    # best_center = best_trial.user_attrs["center"]

    # X_best = X - best_center
    # X_best = X_best[best_ids]
    # Y_best = Y[best_ids]

    # theta_best, _ = get_thetad(X_best.T, dmax)
    # Y_pred = best_trial.user_attrs["coeff"] @ theta_best
    # this_rrmse = rrmse_per_node(Y_best.T, Y_pred)

    best_Ainf = best_trial.user_attrs["Ainf"]
    best_coeff = best_trial.user_attrs["coeff"]

    # print("Mean node-wise RRMSE:", np.mean(this_rrmse))
    import pickle
    import os

    save_dict = {
        "args": vars(args),
        "best_params": study.best_params,
        "best_auc": study.best_value,

        # reconstruction
        "Score": best_score,
        "T_Matrix": T_Matrix,

        # fitting performance
        # "RRMSE": this_rrmse,
        # "Mean_RRMSE": float(np.mean(this_rrmse)),
        # "Y_pred": Y_pred,

        # THIS outputs
        "Ainf": best_Ainf,
        "coeff": best_coeff,

        # "keep_ratio": best_trial.user_attrs["keep_ratio"],
        # "delta": best_trial.user_attrs["delta"],
        # "center": best_trial.user_attrs["center"],
        # "ids": best_trial.user_attrs["ids"],
        # "n_selected": len(best_trial.user_attrs["ids"]),
        # "relerr": best_trial.user_attrs["relerr"],
        #
        # "Y_true_local": Y_best.T,
        # "Y_pred": Y_pred,
    }

    os.makedirs("results", exist_ok=True)

    with open("results/THIS_result_2.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print("Results saved to results/THIS_result_2.pkl")



