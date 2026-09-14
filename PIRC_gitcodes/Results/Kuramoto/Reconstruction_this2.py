import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
import os
from Torch_Library import *
from DateGen_kuramoto import load_or_generate_kuramoto
from Model.this import this, restrict_hypercube_size

def objective_fast(trial, X=None, Y=None, args=None, T=None, ooi=[], dmax=2, ExpandNodes=None):
    lam = trial.suggest_float("threshold", 1e-5, 1, log=True)
    rho = trial.suggest_float("alpha", 1e-5, 1, log=True)
    niter = trial.suggest_categorical("niter", [10])

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

    AUC = ranking_auc_ALL(T_from_Ainf, T)
    AUC = AUC.item() if hasattr(AUC, "item") else float(AUC)

    trial.set_user_attr("Score", T_from_Ainf)

    return float(AUC)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_num', type=int, default=10, help='Number of nodes in the system')
    parser.add_argument('--Pair_strength', type=float, default=0.4, help='Pairwise interaction strength') #1
    parser.add_argument('--Tri_strength', type=float, default=0.4, help='Three-way interaction strength') #1
    parser.add_argument("--N_train", type=int, default=1000)
    parser.add_argument("--N_test", type=int, default=0)
    parser.add_argument("--N_washout", type=int, default=0)
    parser.add_argument("--N_start", type=int, default=0)
    parser.add_argument("--P_probability", type=float, default=0.05) #0.02
    parser.add_argument("--T_probability", type=float, default=0.005)  # 0.02
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()
    set_seed(42)

    ExpandNodes = args.node_num + math.comb(args.node_num - 1, 2)
    ooi = []
    dmax = 2

    with open("results/kuramoto_10_trajectories.pkl", "rb") as f:
        result = pickle.load(f)

    data_list = result["data_list"]
    a2_list = result["a2_list"]
    a3_list = result["a3_list"]
    T_list = result["T_list"]

    Score_list = []
    params_list = []

    X_list = []
    Y_list = []

    AUC_list = []

    for i in range(len(data_list)):
        data=data_list[i][args.N_start: args.N_start+args.N_train+1,]
        T=T_list[i]

        X1 = data[:-1, :]
        X2 = data[1:, :]
        X = X1
        Y = (X2 - X1) / args.dt
        X = X1 % (2 * np.pi)
        X_list.append(X)
        Y_list.append(Y)

        study = optuna.create_study(
            study_name=f"kuramoto",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=int(0.5 * args.n_trials)),
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(lambda trial: objective_fast(trial, X=X, Y=Y, args=args, T=T, ExpandNodes=ExpandNodes), n_trials=args.n_trials)

        best_trial = study.best_trial
        best_score = best_trial.user_attrs["Score"]
        params = study.best_params
        AUC = study.best_value

        Score_list.append(best_score)
        params_list.append(params)
        AUC_list.append(AUC)

    print(np.max(AUC_list))
    print(np.min(AUC_list))
    print(np.mean(AUC_list))

    import pickle
    import os

    save_dict = {
        "params": params_list,
        "AUC": AUC_list,
        "Score": Score_list,
    }

    os.makedirs("results", exist_ok=True)

    with open("results/THIS_result_10.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print("Results saved to results/THIS_result_10.pkl")

