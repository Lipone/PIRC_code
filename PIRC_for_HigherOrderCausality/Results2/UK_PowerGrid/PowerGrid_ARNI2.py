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
    base="polynomial",
    basis_order=2,
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

        llist, cost, vec = ARNI(
            X_expanded_T,
            Y_T,
            base,
            basis_order,
            NODE,
            th=th
        )

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

        coef, *_ = np.linalg.lstsq(
            R.T,
            Y_T[NODE, :],
            rcond=None
        )

        Y_pred[NODE, :] = R.T @ coef

    arni_rrmse = rrmse_per_node(Y_T, Y_pred)

    return arni_rrmse, Y_pred

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
    parser.add_argument("--N_train", type=int, default=1000)
    parser.add_argument("--N_test", type=int, default=0)
    parser.add_argument("--N_washout", type=int, default=0)
    parser.add_argument("--N_start", type=int, default=100)
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

    X = data[:-1, :]
    X = X % (2 * np.pi)
    Y = (data[1:, :]- data[:-1, :])/args.dt
    Y_T = Y.T

    th = 0.0001
    typ="mul"
    basis_order=2
    base = 'fourier'
    T_from_Ainf = np.zeros((args.n, ExpandNodes))
    expanded_cache = {}
    for NODE in range(args.n):
        expanded_cache[(NODE, typ)] = expand_nodes(X.copy(), NODE, type=typ)
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
        for NODE in range(args.n)
    )

    for NODE, vec in results:
        T_from_Ainf[NODE, :] = vec

    AUC = ranking_auc_ALL(T_from_Ainf, T_Matrix)
    AUC = AUC.item() if hasattr(AUC, "item") else float(AUC)

    arni_rrmse, Y_pred = compute_arni_rrmse(

        X,

        Y,

        args,

        base=base,

        basis_order=basis_order,

        expand_type=typ,

        th=th

    )

    print("Mean node-wise RRMSE:", np.mean(arni_rrmse))

    save_dict = {
        "args": vars(args),
        "auc": AUC,

        # reconstruction
        "Score": np.asarray(T_from_Ainf),
        "T_Matrix": T_Matrix,

        # fitting performance
        "RRMSE": arni_rrmse,  # (120,)
        "Mean_RRMSE": float(np.mean(arni_rrmse)),
        "Y_pred": Y_pred  # (120,T)
    }

    os.makedirs("results", exist_ok=True)

    with open("results/ARNI2_result.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print("Results saved to results/ARNI2_result.pkl")