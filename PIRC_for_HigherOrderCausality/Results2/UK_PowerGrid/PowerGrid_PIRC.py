import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
from PIRC import PIRC_flatten as Nonliear_PIRC
from functools import partial
import torch.nn.functional as F
import pandas as pd

def objective_fast(trial, data=None, args=None, T_Matrix=None, search_matrix=None):
    n_units = trial.suggest_categorical('n_units', [2,3,4,5])
    alpha = trial.suggest_float('alpha', 0.1, 1, step=0.1)
    sigma_in = trial.suggest_float('sigma_in', 0.1, 1, step=0.1)
    rho = trial.suggest_float('rho', 0.1, 1, step=0.1)
    tikh = trial.suggest_categorical('tikh', [1,0.1])
    bias = trial.suggest_categorical('bias', [0])
    option = trial.suggest_categorical('option', [3])
    tau = trial.suggest_categorical('tau', [2])
    method = trial.suggest_categorical('method', ['Logistic'])
    connectivity = trial.suggest_categorical('connectivity', [0.5,1.0,0.25,0.75])
    I_type = trial.suggest_categorical('I_type', [1])
    mode = trial.suggest_categorical('mode', [1])
    block_dim = trial.suggest_categorical('block_dim', [2])
    ExpandNodes = args.n + math.comb(args.n - 1, 2)
    device = torch.device("cuda:3")
    T_tensor = torch.as_tensor(T_Matrix, dtype=torch.float32, device=device)

    Score_full = torch.full_like(T_tensor, -1e9)
    X = torch.as_tensor(data, dtype=torch.float32, device=device)
    X_batch = torch.stack(
        [shift_column_to_first(X, i) for i in range(args.n)],
        dim=0
    )
    X_washout, X_train, Y_train, Y_test = split_dataset2_batch(
            X_batch,
            args.N_washout,
            args.N_train,
            args.N_test,
            in_dim=args.n,
            out_dim=args.n
        )
    e_max = 0
    while (e_max == 0):
        Win, Wres = Win_Fixed_Wres(
            ExpandNodes + 1, n_units, device,
            block_dim=block_dim, connectivity=connectivity
        )
        E, _ = torch.linalg.eig(Wres)
        e_max = torch.max(torch.abs(E))
    B = X_batch.shape[0]
    for i in range(B):
        pirc = Nonliear_PIRC(
            n_units=n_units,
            in_dim=args.n,
            out_dim=args.n,
            Win=Win,
            Wres=Wres,
            Expand=1,
            sigma_in=sigma_in,
            rho=rho,
            alpha=alpha,
            tikh=tikh,
            mode=mode,
            block_dim=block_dim,
            I_type=I_type,
            option=option,
            device=device,
            dt=args.dt,
            bias=bias
        )
        R,_,_ = pirc.train(X_washout[i:i + 1, :, :], X_train[i:i + 1, :, :], Y_train[i:i + 1, :, :])
        Y_test_predict = pirc.Prediction2(R, args.N_test, Y_train[i:i + 1, :, :], Y_test[i:i + 1, :, :])
        base_all = Y_test_predict[0, :, :args.N_test, 0]
        others_all = Y_test_predict[2:, :, :args.N_test, 0]
        base_expand = base_all.unsqueeze(0).expand_as(others_all)
        score_EN = Causal_score(
            others_all,
            base_expand,
            method=method,
            tau=tau
        )
        Score = score_EN.transpose(0, 1).contiguous()
        Score_full[i, :] = Score
    Score_full = shift_column_for_Causal_Matrix(Score_full)
    AUC = ranking_auc_ALL(Score_full, T_tensor)

    trial.set_user_attr("Score", Score.detach().cpu().numpy())
    trial.set_user_attr("Win", Win.detach().cpu().numpy())
    trial.set_user_attr("Wres", Wres.detach().cpu().numpy())
    # trial.set_user_attr("total_mse", total_mse.item())
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
    parser.add_argument("--N_test", type=int, default=10)
    parser.add_argument("--N_washout", type=int, default=100)
    parser.add_argument("--N_start", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=500)
    parser.add_argument("--Threshold", type=float, default=1e-3)

    args = parser.parse_args()

    (Xsn, theta, time_point, edges_out, Sd) = read_data(args)
    data=theta.squeeze().T #T*n

    ExpandNodes = args.n + math.comb(args.n - 1, 2)
    set_seed(42)

    data = data[args.N_start:,]

    T_Matrix = np.zeros((args.n, ExpandNodes))
    search_matrix= np.zeros((args.n, args.n))

    for [i, j] in edges_out:
        T_Matrix[int(i), int(j)] = 1
    for [i, j, k] in Sd:
        com_list = list(combinations([x for x in range(args.n) if x != int(i)], 2))
        idx = com_list.index(tuple(sorted((int(j), int(k)))))
        T_Matrix[int(i), args.n + idx] = 1
    print("ground_truth", T_Matrix)

    for [i, j] in edges_out:
        search_matrix[int(i), int(j)] = 1
    for [i, j, k] in Sd:
        search_matrix[int(i), int(j)] = 1
        search_matrix[int(i), int(k)] = 1
    print("search_matrix", search_matrix)
    row_counts = (search_matrix == 1).sum(axis=1)
    print("每一行1的数量:", row_counts)
    print("最大数量:", row_counts.max())
    max_row = row_counts.argmax()
    print("最大所在行:", max_row.item())
    N = search_matrix.shape[0]
    new_matrix = np.zeros_like(search_matrix)
    for i in range(N):
        row = search_matrix[i]
        new_matrix[i] = np.concatenate(([row[i]], row[:i], row[i + 1:]))
    search_matrix_shift = new_matrix
    diff = search_matrix != search_matrix_shift
    print("变化位置（True表示变了）:\n", diff)
    changed_indices = np.argwhere(diff)
    print("变化的索引:", changed_indices)
    count_Pair = np.count_nonzero(T_Matrix[:, :args.n])
    count_Tri = np.count_nonzero(T_Matrix[:, args.n:])

    print("pair:", count_Pair)
    print("Tri:", count_Tri)

    # ========= Optuna =========
    study = optuna.create_study(
        study_name=f"PowerGrid_PIRC",
        direction="maximize",
        sampler = optuna.samplers.TPESampler(n_startup_trials=int(0.5*args.n_trials)),
        pruner=optuna.pruners.MedianPruner()
    )

    callback_with_threshold = partial(stop_when_low_enough, THRESHOLD=args.Threshold)
    study.optimize(lambda trial: objective_fast(trial,data=data,args=args,T_Matrix=T_Matrix,search_matrix=search_matrix_shift), n_trials=args.n_trials, callbacks=[callback_with_threshold])

    best_trial = study.best_trial

    print("Study finished.")
    print("\nBest Hyperparameters:", study.best_params)
    print("\nBest AUC:", study.best_value)
    print("AUC level:", study.best_value)

    best_score = study.best_trial.user_attrs["Score"]
    best_Win = study.best_trial.user_attrs["Win"]
    best_Wres = study.best_trial.user_attrs["Wres"]

    import pickle
    import os

    save_dict = {
        "args": vars(args),
        "best_params": study.best_params,
        "best_auc": study.best_value,
        "Score": best_score,
        "Win": best_Win,
        "Wres": best_Wres,
        "T_Matrix": T_Matrix,
    }

    os.makedirs("results", exist_ok=True)

    with open("results/PIRC_result_0823.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print("Results saved to results/PIRC_result_0823.pkl")




