import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
from PIRC import PIRC_flatten as Nonliear_PIRC
from Model.this import *
from scipy.sparse import csc_matrix
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
        Wres[:n1, :n1].uniform_(-1, 1)
        Wres[n1:, n1:].uniform_(-1, 1)
        d1 = block_dim // 2
        Win = torch.zeros(
            ExpandNodes, block_dim, n_units,
            device=device
        )
        Win[:, :d1, :n1].uniform_(-1, 1)
        Win[:, d1:, n1:].uniform_(-1, 1)
    elif connectivity == -0.5:
        n1 = n_units // 2
        Wres = torch.zeros(
            n_units, n_units,
            device=device
        )
        Wres[:, :].uniform_(-1, 1)
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

def objective_fast(trial, args=None, X_all=None, Y_all=None,device=None, selected=None):
    n_units = trial.suggest_categorical('n_units', [5,10])
    alpha = trial.suggest_float('alpha', 0.1, 1, step=0.1)
    sigma_in = trial.suggest_float('sigma_in', 0.1, 1, step=0.1)
    rho = trial.suggest_float('rho', 0.1, 1, step=0.1)
    tikh = trial.suggest_categorical('tikh', [1,0.1])
    option = trial.suggest_categorical('option', [3])
    tau = trial.suggest_categorical('tau', [2])
    method = trial.suggest_categorical('method', ['Logistic'])
    connectivity = trial.suggest_categorical('connectivity', [1.0,0.5,-0.5])
    mode = trial.suggest_categorical('mode', [2])
    block_dim = trial.suggest_categorical('block_dim', [2])
    bias = trial.suggest_categorical('bias', [0])
    ExpandNodes = args.n + math.comb(args.n - 1, 2)
    Win, Wres = Win_Fixed_Wres(
        ExpandNodes + 1, n_units, device,
        block_dim=block_dim, connectivity=connectivity
    )
    Loss=[]
    for item in selected:
        X = item["X"]
        X = torch.as_tensor(X, device=device, dtype=torch.float32).T
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
            I_type=args.I_type,
            option=option,
            device=device,
            dt= 1 / 160,
            bias=bias
        )
        R, _ = pirc.train(X_washout, X_train, Y_train)
        if mode == 1:
            Y_test_predict = pirc.Prediction(R, args.N_test, Y_train, Y_test)
        else:
            Y_test_predict = pirc.Prediction2(R, args.N_test, Y_train, Y_test)
        base_all = Y_test_predict[0, :, :args.N_test, 0]  # (B, T_test)
        target = Y_test[:, :, 0]  # (B, T_test)
        total_mse = (base_all - target).pow(2).mean()
        Loss.append(total_mse)

        trial.set_user_attr("Win", Win.detach().cpu().numpy())
        trial.set_user_attr("Wres", Wres.detach().cpu().numpy())

    return torch.stack(Loss).mean().item()

def gridSearch_PIRC(args):

    device = torch.device("cuda:2")

    T = args.N_train + args.N_test + args.N_washout + args.N_start + 1

    nz = 7  # scalp zones
    subjects = list_all_subjects(109)
    states = ["01", "02"]  # resting
    s = np.loadtxt(f"../../EEG_data/eeg-data/sensors-{nz}.csv", dtype=str, delimiter=",")
    z = np.loadtxt(f"../../EEG_data/eeg-data/zones-{nz}.csv", dtype=int, delimiter=",")
    s2z = {s[i]: z[i] for i in range(len(s))}

    dataset = []
    for su in subjects:
        for st in states:
            key = f"S{su}R{st}"
            file = f"../../EEG_data/physionet.org/files/eegmmidb/1.0.0/S{su}/{key}.edf"
            s2signal = read_eeg(file)
            asig = average_over_zones(s2signal, s2z)

            tmp = np.max(np.abs(asig), axis=0)
            valid_idx = np.where(tmp > 1e-6)[0]
            if len(valid_idx) == 0:
                continue
            if np.all(np.diff(valid_idx) == 1):
                pass
            else:
                print("不连续")
            start, end = valid_idx[0], valid_idx[-1]

            X0 = denoise_fourier(asig[:, start:end + 1], 100)
            if args.norm==1:
                X0 = X0 / np.mean(np.abs(X0))
            else:
                X0 = (X0 - X0.mean(axis=1, keepdims=True)) / (X0.std(axis=1, keepdims=True) + 1e-12)
            X = X0[:,:T]
            dataset.append({
                "key": key,
                "X": X,
            })

    num_select = 20
    set_seed(42)
    selected = random.sample(dataset, min(num_select, len(dataset)))
    # ========= Optuna =========
    study = optuna.create_study(
        study_name=f"EEG_PIRC",
        direction="minimize",
        sampler = optuna.samplers.TPESampler(n_startup_trials=int(0.5*args.n_trials)),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(lambda trial: objective_fast(trial,args=args, device=device, selected=selected), n_trials=args.n_trials)

    print("Study finished.")
    print("\nBest Hyperparameters:", study.best_params)
    print("\nLowest Loss:", study.best_value)

    best_trial = study.best_trial
    best_Win = best_trial.user_attrs["Win"]
    best_Wres = best_trial.user_attrs["Wres"]

    return study.best_params,best_Win, best_Wres

    # import pickle
    #
    # save_dict = {
    #     "params": study.best_params,
    #     "loss": study.best_value,
    #     "Win": best_Win,
    #     "Wres": best_Wres,
    # }
    #
    # with open("EEG_PIRC_best2.pkl", "wb") as f:
    #     pickle.dump(save_dict, f)



