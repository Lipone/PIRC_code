from PIRC import PIRC_flatten as Nonliear_PIRC
from Data_gen import *
import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *

# def Win_Fixed_Wres(
#     ExpandNodes,
#     n_units,
#     device,
#     block_dim=1,
#     connectivity=1.0
# ):
#     if connectivity == 1.0:
#         Win = 2 * torch.rand(
#             ExpandNodes, block_dim, n_units,
#             device=device
#         ) - 1
#
#         Wres = torch.empty(
#             n_units, n_units,
#             device=device
#         ).uniform_(-1, 1)
#     elif connectivity == 0.5:
#         n1 = n_units // 2
#         Wres = torch.zeros(
#             n_units, n_units,
#             device=device
#         )
#         Wres[:n1, :n1].uniform_(-1, 1)
#         Wres[n1:, n1:].uniform_(-1, 1)
#         d1 = block_dim // 2
#         Win = torch.zeros(
#             ExpandNodes, block_dim, n_units,
#             device=device
#         )
#         Win[:, :d1, :n1].uniform_(-1, 1)
#         Win[:, d1:, n1:].uniform_(-1, 1)
#     elif connectivity == 0.25:
#         n1 = n_units // 2
#         Wres = torch.zeros(
#             n_units, n_units,
#             device=device
#         )
#         # Wres[:n1, :n1].uniform_(-1, 1)
#         Wres[n1:, n1:].uniform_(-1, 1)
#         d1 = block_dim // 2
#         Win = torch.zeros(
#             ExpandNodes, block_dim, n_units,
#             device=device
#         )
#         # Win[:, :d1, :n1].uniform_(-1, 1)
#         Win[:, d1:, n1:].uniform_(-1, 1)
#     elif connectivity == 0.75:
#         n1 = n_units // 2
#         Wres = torch.zeros(
#             n_units, n_units,
#             device=device
#         )
#         # Wres[:n1, :n1].uniform_(-1, 1)
#         Wres[n1:, n1:].uniform_(-1, 1)
#         d1 = block_dim // 2
#         Win = torch.zeros(
#             ExpandNodes, block_dim, n_units,
#             device=device
#         )
#         # Win[:, :d1, :n1].uniform_(-1, 1)
#         Win[:, d1:, n1:].uniform_(-1, 1)
#     return Win, Wres
def Win_Fixed_Wres(
    ExpandNodes,
    n_units,
    device,
    block_dim=1,
    connectivity=1.0
):
    if connectivity == 1.0:
        Win = torch.zeros(
            ExpandNodes, block_dim, n_units,
            device=device
        )
        # 只生成一套非零位置
        idx = torch.randint(
            0, block_dim,
            (1, 1, n_units),
            device=device
        )
        # 所有 ExpandNodes 使用相同位置
        idx = idx.expand(ExpandNodes, -1, -1)
        # 但非零值可以各自随机
        values = 2 * torch.rand(
            ExpandNodes, 1, n_units,
            device=device
        ) - 1
        Win.scatter_(1, idx, values)
        Wres = torch.empty(
            n_units, n_units,
            device=device
        ).uniform_(-1, 1)
    if connectivity == -1.0:
        Win = torch.zeros(
            ExpandNodes, block_dim, n_units,
            device=device
        )

        # 每一列随机选择一个非零位置
        idx = torch.randint(
            0, block_dim,
            (ExpandNodes, 1, n_units),
            device=device
        )

        # 非零值随机取自 [-1, 1]
        values = 2 * torch.rand(
            ExpandNodes, 1, n_units,
            device=device
        ) - 1

        Win.scatter_(1, idx, values)

        Wres = torch.empty(
            n_units, n_units,
            device=device
        ).uniform_(-1, 1)
    elif connectivity == 0.5:
        n1 = n_units // 2
        Wres = torch.zeros(
            n_units, n_units,
            device=device
        )
        Wres[:, :].uniform_(-1, 1)
        # Wres[n1:, n1:].uniform_(-1, 1)
        d1 = block_dim // 2
        Win = torch.zeros(
            ExpandNodes, block_dim, n_units,
            device=device
        )
        Win[:, :d1, :n1].uniform_(-1, 1)
        Win[:, d1:, n1:].uniform_(-1, 1)
    elif connectivity == 0.25:
        # n1 = n_units // 2
        Wres = torch.zeros(
            n_units, n_units,
            device=device
        )
        Wres[:, :].uniform_(-1, 1)
        # Wres[n1:, n1:].uniform_(-1, 1)
        d1 = block_dim // 2
        Win = torch.zeros(
            ExpandNodes, block_dim, n_units,
            device=device
        )
        # Win[:, :d1, :n1].uniform_(-1, 1)
        Win[:, d1:, :].uniform_(-1, 1)
    elif connectivity == 0.75:
        # n1 = n_units // 2
        Wres = torch.zeros(
            n_units, n_units,
            device=device
        )
        Wres[:, :].uniform_(-1, 1)
        # Wres[n1:, n1:].uniform_(-1, 1)
        d1 = block_dim // 2
        Win = torch.zeros(
            ExpandNodes, block_dim, n_units,
            device=device
        )
        Win[:, :d1, :].uniform_(-1, 1)
        # Win[:, d1:, n1:].uniform_(-1, 1)
    return Win, Wres

def Causal_score(y_pred, y_true, method="exp", tau=2, eps=1e-12):

    # 转成 tensor
    y_pred = torch.as_tensor(y_pred)
    y_true = torch.as_tensor(y_true)

    # mse over time dimension only
    mse = (y_pred - y_true).pow(2).mean(dim=-1).abs()         # (...,)

    # scale: mean over time of y_true (同你原实现)
    # scale = y_true.mean(dim=-1).abs().clamp_min(eps)          # (...,)
    scale1 = y_true.pow(2).mean(dim=-1).abs()
    scale2 = y_true.var(dim=-1, correction=0)# (...,)

    return torch.sqrt(mse / scale1)
    return torch.sqrt(mse / scale2)

def objective_fast(trial, data=None, args=None, T_Matrix=None):
    # ====== Hyperparameters ======
    n_units = trial.suggest_categorical('n_units', [5,10])
    alpha = trial.suggest_float('alpha', 0.1, 1, step=0.1)
    sigma_in = trial.suggest_float('sigma_in', 0.1, 1, step=0.1)
    rho = trial.suggest_float('rho', 0.1, 1, step=0.1)
    tikh = trial.suggest_categorical('tikh', [1.0,1e-1])
    option = trial.suggest_categorical('option', [3])
    tau = trial.suggest_categorical('tau', [2]) #在auc尺度下tau的选择几乎不影响结果，可以固定以加快搜索
    method = trial.suggest_categorical('method', ['Logistic']) #['exp', 'Logistic', 'power']
    connectivity = trial.suggest_categorical('connectivity', [1.0,0.5])
    I_type = trial.suggest_categorical('I_type', [1])
    block_dim = trial.suggest_categorical('block_dim', [2])
    ExpandNodes = (args.node_num + math.comb(args.node_num - 1, 2)+1) #10
    device = torch.device("cuda:3")

    set_seed(42)
    Win, Wres = Win_Fixed_Wres(
        ExpandNodes, n_units, device,
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
        sigma_in=sigma_in, rho=rho, alpha=alpha, tikh=tikh, bias=0, dt=args.dt,
        device=device, block_dim=block_dim, I_type=I_type, option=option
    )

    R, _ = pirc.train(X_washout, X_train, Y_train)

    Y_test_predict = pirc.Prediction2(R, args.N_test, Y_train, Y_test)

    base_all = Y_test_predict[0, :, :args.N_test, 0]  # base: (B, T)
    others_all = Y_test_predict[2: , :, :args.N_test, 0]  # others: (E, B, T)
    total_mse =  ((Y_test[0,:,:].T-base_all).pow(2).sum())/(base_all.shape[0]*base_all.shape[1])
    base_expand = base_all.unsqueeze(0).expand_as(others_all)  # (E, B, T)
    score_EN = Causal_score(
        others_all,
        base_expand,
        method=method,
        tau=tau
    )
    Score = score_EN.transpose(0, 1).contiguous()  # (B, E)
    Score = shift_column_for_Causal_Matrix(Score)
    Score.fill_diagonal_(1.0)
    AUC = ranking_auc_ALL(Score, T_tensor)

    diag_mask = torch.zeros_like(T_tensor, dtype=torch.bool)
    diag_mask.fill_diagonal_(True)
    mask_1 = (T_tensor == 1) & (~diag_mask)
    mask_0 = (T_tensor == 0) & (~diag_mask)
    score_1 = Score[mask_1]
    score_0 = Score[mask_0]
    causal_deviation=(score_1.min().item()-score_0.max().item())/(score_1.max().item()-score_0.min().item())
    noncausal_deviation=(score_0.max().item()-score_0.min().item())/(score_1.max().item()-score_0.min().item())

    target = Score[1, 0]  # 第二行最后一个值
    # row = Score[1]
    # mask = torch.ones_like(row, dtype=torch.bool)
    # mask[1] = False  # 排除第二行的对角元素
    # mask[0] = False  # 排除 target 自己
    # max_other = row[mask].max()
    # penalty = torch.relu(max_other - target)

    penalty =torch.relu(score_1.max().item()- target)

    print(f"AUC: { AUC}, MSE: {total_mse}")
    return total_mse+(1-AUC)*10+1-causal_deviation+noncausal_deviation+penalty*1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_num', type=int, default=3, help='Number of nodes in the system')
    parser.add_argument("--N_train", type=int, default=2000)
    parser.add_argument("--N_test", type=int, default=100)
    parser.add_argument("--N_washout", type=int, default=0)
    parser.add_argument("--N_start", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--use_sin", type=float, default=0)
    parser.add_argument("--Threshold", type=float, default=1e-3)
    args = parser.parse_args()
    ExpandNodes = args.node_num + math.comb(args.node_num - 1, 2)
    set_seed(42)
    t_all=(args.N_train+args.N_test+args.N_washout+args.N_start+1)*args.dt
    # data generation
    t, xyz = generate_lorenz_data(
        t_span=(0, t_all),
        dt=args.dt,
        initial_state=(1.0, 1.0, 1.0),
        sigma=10.0,
        rho=28.0,
        beta=8 / 3
    )
    X = xyz.T  # (N, 3)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # normalization

    #ground Truth matrix
    T_Matrix=[
        [1,1,0,0],
        [1,1,0,1],
        [0,0,1,1]
    ]

    # ========= Optuna =========
    study = optuna.create_study(
        study_name=f"Lorenz_PIRC_flatten_enhance",
        direction="minimize",
        sampler = optuna.samplers.TPESampler(n_startup_trials=int(0.9*args.n_trials)),
        pruner=optuna.pruners.MedianPruner()
    )

    # callback_with_threshold = partial(stop_when_low_enough, THRESHOLD=args.Threshold)
    study.optimize(lambda trial: objective_fast(trial,data=X,args=args,T_Matrix=T_Matrix), n_trials=args.n_trials)

    print("Study finished.")
    print("\nBest Hyperparameters:", study.best_params)
    print("\nLowest Loss:", study.best_value)

# Best Hyperparameters: {'n_units': 10, 'alpha': 0.4, 'sigma_in': 0.30000000000000004, 'rho': 1.0, 'tikh': 1.0, 'option': 2, 'tau': 2, 'method': 'Logistic', 'connectivity': 1.0, 'I_type': 1, 'mode': 2, 'block_dim': 1, 'Base': 'prediction'}
#
# Lowest Loss: 7.094440661603585e-05
# Best Hyperparameters: {'n_units': 10, 'alpha': 1.0, 'sigma_in': 0.2, 'rho': 0.1, 'tikh': 0.1, 'option': 2, 'tau': 2, 'method': 'Logistic', 'connectivity': 1.0, 'I_type': 1, 'mode': 2, 'block_dim': 1, 'Base': 'prediction'}
#
# Lowest Loss: 0.00010138809739146382
# Best Hyperparameters: {'n_units': 10, 'alpha': 0.7000000000000001, 'sigma_in': 0.2, 'rho': 0.1, 'tikh': 1.0, 'option': 2, 'tau': 2, 'method': 'Logistic', 'connectivity': 1.0, 'I_type': 1, 'mode': 2, 'block_dim': 1, 'Base': 'prediction'}
#
# Lowest Loss: 0.000140117816044949
