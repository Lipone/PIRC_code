import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
import os
from Torch_Library import *
from Model.ARNI import ARNI, expand_nodes
from joblib import Parallel, delayed
from DateGen_kuramoto import load_or_generate_kuramoto

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

def objective_fast(trial, X=None, Y=None, args=None, T_Matrix=None, base='fourier', ExpandNodes=None):
    th = trial.suggest_categorical("th", [0.00001,0.0001, 0.001, 0.01, 0.1])
    # th=0.0001
    type = trial.suggest_categorical("type", ["mul"])
    basis_order = trial.suggest_categorical("basis_order", [2])

    T_from_Ainf = np.zeros((args.node_num, ExpandNodes))

    Y_T = Y.T

    expanded_cache = {}
    for NODE in range(args.node_num):
        expanded_cache[(NODE, type)] = expand_nodes(X.copy(), NODE, type=type)

    results = Parallel(n_jobs=-1)(
        delayed(run_node)(
            NODE,
            type,
            base,
            basis_order,
            expanded_cache,
            Y_T,
            th=th
        )
        for NODE in range(args.node_num)
    )

    for NODE, vec in results:
        T_from_Ainf[NODE, :] = vec

    AUC = ranking_auc_ALL(T_from_Ainf, T_Matrix)
    AUC = AUC.item() if hasattr(AUC, "item") else float(AUC)

    trial.set_user_attr("Score", np.asarray(T_from_Ainf))

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
    parser.add_argument("--n_trials", type=int, default=2)
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
    X = (X1 + X2) * 0.5
    X = X % (2 * np.pi)
    Y = (data[1:, :] - data[:-1, :]) / args.dt

    study = optuna.create_study(
        study_name=f"Kuramoto_ARNI",
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
    }

    os.makedirs("results", exist_ok=True)

    with open("results/ARNI2_result.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print("Results saved to results/ARNI2_result.pkl")
