import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
import os
from Torch_Library import *
from Model.ARNI import ARNI, expand_nodes
from joblib import Parallel, delayed
from DateGen_kuramoto import load_or_generate_kuramoto

def run_node(NODE, typ, base, order, expanded_cache, Y_T, th=1e-4):
    try:
        X_expanded = expanded_cache[(NODE, typ)]
        llist, cost, vec = ARNI(
            X_expanded.T,
            Y_T,
            base,
            order,
            NODE,
            th=th
        )
        if not np.all(np.isfinite(vec)):
            vec = np.zeros(X_expanded.shape[1])
        return NODE, vec
    except np.linalg.LinAlgError:
        X_expanded = expanded_cache[(NODE, typ)]
        return NODE, np.zeros(X_expanded.shape[1])
    except Exception as e:
        X_expanded = expanded_cache[(NODE, typ)]
        print(f"[WARN] NODE={NODE} failed: {e}")
        return NODE, np.zeros(X_expanded.shape[1])

def objective_fast(
    trial,
    X=None,
    Y=None,
    args=None,
    T=None,
    base=None,
    ExpandNodes=None,
):
    th = trial.suggest_float("th", 1e-5, 1, log=True)
    typ = trial.suggest_categorical("type", ["mul", "add"])
    basis_order = trial.suggest_categorical("basis_order", [2,3,4,5])

    T_from_Ainf = np.zeros((args.node_num, ExpandNodes))
    Y_T = Y.T

    expanded_cache = {}
    for NODE in range(args.node_num):
        expanded_cache[(NODE, typ)] = expand_nodes(
            X.copy(),
            NODE,
            type=typ
        )

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
        for NODE in range(args.node_num)
    )

    for NODE, vec in results:
        T_from_Ainf[NODE, :] = vec

    AUC = ranking_auc_ALL(T_from_Ainf, T)
    AUC = AUC.item() if hasattr(AUC, "item") else float(AUC)


    trial.set_user_attr("Score", np.asarray(T_from_Ainf))

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
        data=data_list[i][args.N_start:,]
        T=T_list[i]

        X1 = data[:-1, :]
        X2 = data[1:, :]
        X = X1
        Y = (X2 - X1) / args.dt
        X = X1 % (2 * np.pi)
        X_list.append(X)
        Y_list.append(Y)

        study = optuna.create_study(
            study_name=f"Kuramoto_ARNI2",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=int(0.5 * args.n_trials)),
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(
            lambda trial: objective_fast(
                trial,
                X=X,
                Y=Y,
                args=args,
                T=T,
                base="fourier",
                ExpandNodes=ExpandNodes
            ),
            n_trials=args.n_trials
        )

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

    with open("results/ARNI2_10_result.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print("Results saved to results/ARNI2_10_result.pkl")
