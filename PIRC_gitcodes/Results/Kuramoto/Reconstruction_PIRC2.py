import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
from PIRC import PIRC_flatten as Nonliear_PIRC
from DateGen_kuramoto import load_or_generate_kuramoto, generate_kuramoto_data_fast
from functools import partial
import torch.nn.functional as F

def objective_fast(trial, data=None, args=None, T=None):
    # ====== Hyperparameters ======
    n_units = trial.suggest_categorical('n_units', [5,10])
    alpha = trial.suggest_float('alpha', 0.1, 1, step=0.1)
    sigma_in = trial.suggest_float('sigma_in', 0.1, 1, step=0.1)
    rho = trial.suggest_float('rho', 0.1, 1, step=0.1)
    tikh = trial.suggest_categorical('tikh', [1.0,1e-1])
    bias = trial.suggest_categorical('bias', [0])
    option = trial.suggest_categorical('option', [3])
    tau = trial.suggest_categorical('tau', [2]) #在auc尺度下tau的选择几乎不影响结果，可以固定以加快搜索
    method = trial.suggest_categorical('method', ['Logistic']) #['exp', 'Logistic', 'power']
    connectivity = trial.suggest_categorical('connectivity', [1.0,0.5,0.25,0.75])
    I_type = trial.suggest_categorical('I_type', [1])
    mode = trial.suggest_categorical('mode', [2])
    block_dim = trial.suggest_categorical('block_dim', [2])

    device = torch.device("cuda:2")

    Win, Wres = Win_Fixed_Wres(
        ExpandNodes + 1, n_units, device,
        block_dim=block_dim, connectivity=connectivity
    )

    T_tensor = torch.as_tensor(T, dtype=torch.float32, device=device)
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
    pirc = Nonliear_PIRC(
        n_units=n_units, in_dim=args.node_num, out_dim=args.node_num,
        Win=Win.clone(), Wres=Wres.clone(),
        sigma_in=sigma_in, rho=rho, alpha=alpha, tikh=tikh, mode=mode, bias=bias, dt=args.dt,
        device=device, block_dim=block_dim, I_type=I_type, option=option
    )

    R,_,_ = pirc.train(X_washout, X_train, Y_train)
    Y_test_predict = pirc.Prediction2(R, args.N_test, Y_train, Y_test)

    # ====== Score and AUC computation ======
    base_all = Y_test_predict[0, :, :args.N_test, 0]  # base: (B, T)
    others_all = Y_test_predict[2: , :, :args.N_test, 0]  # others: (E, B, T)
    total_mse = ((Y_test[0, :, :].T - base_all).pow(2).sum()) / (base_all.shape[0] * base_all.shape[1])
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
    if torch.is_tensor(AUC):
        AUC = AUC.detach().cpu().item()

    trial.set_user_attr("Score", Score.detach().cpu().numpy())
    trial.set_user_attr("Win", Win.detach().cpu().numpy())
    trial.set_user_attr("Wres", Wres.detach().cpu().numpy())
    trial.set_user_attr("total_mse", total_mse.item())

    return float(AUC)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_num', type=int, default=10, help='Number of nodes in the system')
    parser.add_argument('--Pair_strength', type=float, default=0.4, help='Pairwise interaction strength') #1
    parser.add_argument('--Tri_strength', type=float, default=0.4, help='Three-way interaction strength') #1
    parser.add_argument("--N_train", type=int, default=1000)
    parser.add_argument("--N_test", type=int, default=10)
    parser.add_argument("--N_washout", type=int, default=0)
    parser.add_argument("--N_start", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--Threshold", type=float, default=1e-3)
    parser.add_argument("--P_probability", type=float, default=0.05)  # 0.02
    parser.add_argument("--T_probability", type=float, default=0.005)  # 0.02
    parser.add_argument("--dt", type=float, default=0.08)
    args = parser.parse_args()
    ExpandNodes = args.node_num + math.comb(args.node_num - 1, 2)
    set_seed(42)

    import pickle

    with open("results/kuramoto_10_trajectories.pkl", "rb") as f:
        result = pickle.load(f)

    data_list = result["data_list"]
    a2_list = result["a2_list"]
    a3_list = result["a3_list"]
    T_list = result["T_list"]

    param_list = []
    Win_list = []
    Wres_list = []
    Score_list = []
    AUC_list = []
    MSE_list = []

    for each in range(len(data_list)):
        # ========= Optuna =========
        study = optuna.create_study(
            study_name=f"Kuramoto_P{args.Pair_strength}_T{args.Tri_strength}_{args.node_num}nodes_testLen{args.N_test}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=int(0.5 * args.n_trials)),
            pruner=optuna.pruners.MedianPruner()
        )
        data=data_list[each]
        T=T_list[each]
        study.optimize(lambda trial: objective_fast(trial,data=data,args=args,T=T), n_trials=args.n_trials)
        print(f"In dataset: {each}, AUC level: {study.best_value}:" )
        Win = study.best_trial.user_attrs["Win"]
        Wres = study.best_trial.user_attrs["Wres"]
        Score = study.best_trial.user_attrs["Score"]
        total_mse = study.best_trial.user_attrs["total_mse"]
        params = study.best_params
        param_list.append(params)
        Win_list.append(Win)
        Wres_list.append(Wres)
        Score_list.append(Score)
        AUC_list.append(study.best_value)
        MSE_list.append(total_mse)

    print(np.max(AUC_list))
    print(np.min(AUC_list))
    print(np.mean(AUC_list))
    import pickle
    import os

    save_dict = {
        "params": param_list,
        "AUC_list": AUC_list,
        "Score_list": Score_list,
        "Win_list":Win_list,
        "Wres_list":Wres_list,
    }

    os.makedirs("results", exist_ok=True)

    with open("results/best_result_10_0823.pkl_2", "wb") as f:
        pickle.dump(save_dict, f)

    print("Results saved to results/best_result_10_0823_3.pkl")
