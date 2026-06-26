import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
from Model.PIRC import PIRC_flatten as Nonliear_PIRC
from DateGen_kuramoto import load_or_generate_kuramoto
from functools import partial
import torch.nn.functional as F

def objective_fast(trial, data=None, args=None, T_Matrix=None):
    # ====== Hyperparameters ======
    n_units = trial.suggest_categorical('n_units', [5,10])
    alpha = trial.suggest_float('alpha', 0.1, 1, step=0.1)
    sigma_in = trial.suggest_float('sigma_in', 0.1, 1, step=0.1)
    rho = trial.suggest_float('rho', 0.1, 1, step=0.1)
    tikh = trial.suggest_categorical('tikh', [1.0,1e-1])
    bias = trial.suggest_categorical('bias', [0,1])
    option = trial.suggest_categorical('option', [2])
    tau = trial.suggest_categorical('tau', [2]) #在auc尺度下tau的选择几乎不影响结果，可以固定以加快搜索
    method = trial.suggest_categorical('method', ['Logistic']) #['exp', 'Logistic', 'power']
    connectivity = trial.suggest_categorical('connectivity', [1.0])
    I_type = trial.suggest_categorical('I_type', [1])
    mode = trial.suggest_categorical('mode', [1])
    block_dim = trial.suggest_categorical('block_dim', [1])
    Base = trial.suggest_categorical('Base', ['prediction'])

    device = torch.device("cuda:3")

    Win, Wres = Random_Win_Fixed_Wres(
        ExpandNodes+1, n_units, device,
        block_dim=block_dim, connectivity=connectivity
    )

    X = torch.as_tensor(data, dtype=torch.float32, device=device)
    X_batch = torch.stack(
        [shift_column_to_first(X, i) for i in range(args.node_num)],
        dim=0
    )
    X_washout, X_train, Y_train, Y_test = split_dataset2_batch(
        X_batch,
        args.N_washout,
        args.N_train,
        args.N_test,
        in_dim=args.node_num,
        out_dim=args.node_num
    )

    T_tensor = torch.as_tensor(T_Matrix, dtype=torch.float32, device=device)

    pirc = Nonliear_PIRC(
        n_units=n_units, in_dim=args.node_num, out_dim=args.node_num,
        Win=Win, Wres=Wres,
        sigma_in=sigma_in, rho=rho, alpha=alpha, tikh=tikh, mode=mode, bias=bias, dt=args.dt,
        device=device, block_dim=block_dim, I_type=I_type, option=option
    )

    R,_ = pirc.train(X_washout, X_train, Y_train)
    Y_test_predict = pirc.Prediction(R, args.N_test, Y_train, Y_test)

    # ====== Score and AUC computation ======
    base_all = Y_test_predict[0, :, :args.N_test, 0]  # base: (B, T)
    others_all = Y_test_predict[2: , :, :args.N_test, 0]  # others: (E, B, T)
    base_expand = base_all.unsqueeze(0).expand_as(others_all)  # (E, B, T)
    score_EN = score_0_1_torch(
        others_all,
        base_expand,
        method=method,
        tau=tau
    )
    Score = score_EN.transpose(0, 1).contiguous()  # (B, E)
    Score = shift_column_for_Causal_Matrix(Score)
    Score.fill_diagonal_(1.0)
    AUC = ranking_auc_ALL(Score, T_tensor)

    return AUC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_num', type=int, default=20, help='Number of nodes in the system')
    parser.add_argument('--Pair_strength', type=float, default=0.4, help='Pairwise interaction strength') #1
    parser.add_argument('--Tri_strength', type=float, default=0.4, help='Three-way interaction strength') #1
    parser.add_argument("--N_train", type=int, default=10000)
    parser.add_argument("--N_test", type=int, default=10)
    parser.add_argument("--N_washout", type=int, default=100)
    parser.add_argument("--N_start", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=1000)
    parser.add_argument("--Threshold", type=float, default=1e-3)
    parser.add_argument("--P_probability", type=float, default=0.05)  # 0.02
    parser.add_argument("--T_probability", type=float, default=0.005)  # 0.02
    parser.add_argument("--dt", type=float, default=0.08)
    args = parser.parse_args()
    ExpandNodes = args.node_num + math.comb(args.node_num - 1, 2)
    set_seed(42)

    data, a2, a3 = load_or_generate_kuramoto(args)
    data = data[args.N_start:,]

    import numpy as np
    import matplotlib.pyplot as plt

    theta = data  # shape: (T, N)，你的 kuramoto 生成的是相位

    # 1. Kuramoto order parameter
    R = np.abs(np.mean(np.exp(1j * theta), axis=1))

    print("mean R last half:", np.mean(R[len(R) // 2:]))
    print("max R:", np.max(R))

    plt.plot(R)
    plt.xlabel("time step")
    plt.ylabel("R(t)")
    plt.show()

    theta_tail = theta[len(theta) // 2:]

    dtheta = np.angle(
        np.exp(1j * (theta_tail[:, :, None] - theta_tail[:, None, :]))
    )

    phase_diff_std = np.mean(np.std(dtheta, axis=0))

    print("mean phase-difference std:", phase_diff_std)

    T_Matrix=np.zeros((args.node_num, ExpandNodes))
    T_Matrix[:, :args.node_num] = (a2 != 0).astype(int)
    for i in range(args.node_num):
        com_list=list(combinations([x for x in range(args.node_num) if x != i], 2))
        for index,(j,k) in enumerate(com_list):
            T_Matrix[i, args.node_num+index] = int(a3[i][j][k] != 0)
    print("ground_truth", T_Matrix)
    count_Pair = np.count_nonzero(T_Matrix[:, :args.node_num])
    count_Tri = np.count_nonzero(T_Matrix[:, args.node_num:])
    print("pair:", count_Pair)
    print("Tri:", count_Tri)

    # ========= Optuna =========
    study = optuna.create_study(
        study_name=f"Kuramoto_P{args.Pair_strength}_T{args.Tri_strength}_{args.node_num}nodes_testLen{args.N_test}",
        direction="maximize",
        sampler = optuna.samplers.TPESampler(n_startup_trials=int(0.8*args.n_trials)),
        pruner=optuna.pruners.MedianPruner()
    )

    callback_with_threshold = partial(stop_when_low_enough, THRESHOLD=args.Threshold)
    study.optimize(lambda trial: objective_fast(trial,data=data,args=args,T_Matrix=T_Matrix), n_trials=args.n_trials, callbacks=[callback_with_threshold])

    print("Study finished.")
    print("\nBest Hyperparameters:", study.best_params)
    print("\nbest_AUC:", study.best_value)
    print("AUC level:", study.best_value)


# Best Hyperparameters: {'n_units': 5, 'alpha': 1.0, 'sigma_in': 1.0, 'rho': 0.6, 'tikh': 1.0, 'bias': 0, 'option': 2, 'tau': 2, 'method': 'Logistic', 'connectivity': 1.0, 'I_type': 1, 'mode': 1, 'block_dim': 2, 'Base': 'prediction'}
#
# best_AUC: 0.8426504135131836
# AUC level: 0.8426504135131836

# Best Hyperparameters: {'n_units': 5, 'alpha': 1.0, 'sigma_in': 1.0, 'rho': 0.7000000000000001, 'tikh': 0.1, 'bias': 0, 'option': 2, 'tau': 2, 'method': 'Logistic', 'connectivity': 1.0, 'I_type': 1, 'mode': 1, 'block_dim': 2, 'Base': 'prediction'}
#
# best_AUC: 0.8513253927230835
# AUC level: 0.8513253927230835

# Best Hyperparameters: {'n_units': 10, 'alpha': 0.8, 'sigma_in': 1.0, 'rho': 1.0, 'tikh': 0.1, 'bias': 0, 'option': 2, 'tau': 2, 'method': 'Logistic', 'connectivity': 1.0, 'I_type': 1, 'mode': 1, 'block_dim': 2, 'Base': 'prediction'}
#
# best_AUC: 0.854686975479126
# AUC level: 0.854686975479126

# Trial 316 finished with value: 0.8814795017242432 and parameters: {'n_units': 5, 'alpha': 0.8, 'sigma_in': 0.5, 'rho': 0.30000000000000004, 'tikh': 1.0, 'bias': 0, 'option': 2, 'tau': 2, 'method': 'Logistic', 'connectivity': 1.0, 'I_type': 1, 'mode': 1, 'block_dim': 1, 'Base': 'prediction'}. Best is trial 316 with value: 0.8814795017242432.

# Best Hyperparameters: {'n_units': 5, 'alpha': 1.0, 'sigma_in': 0.4, 'rho': 0.1, 'tikh': 1.0, 'bias': 0, 'option': 2, 'tau': 2, 'method': 'Logistic', 'connectivity': 1.0, 'I_type': 1, 'mode': 1, 'block_dim': 1, 'Base': 'prediction'}
#
# best_AUC: 0.8830809593200684
# AUC level: 0.8830809593200684