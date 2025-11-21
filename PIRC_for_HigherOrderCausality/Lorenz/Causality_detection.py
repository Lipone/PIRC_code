import optuna
import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Data_gen import * 
from Model.PIRC import PIRC
from Model.Causal_Index import cal_CausalIndex_PIRC
from Library import *
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed



# 获取逻辑核心总数
total_cores = os.cpu_count()
# 设置合理的核心使用比例（例如 50%）
n_jobs = max(1, total_cores // 4)

#data generation
t, xyz = generate_lorenz_data(
        t_span=(0, 1000),
        dt=0.1,
        initial_state=(1.0, 1.0, 1.0),
        sigma=10.0,
        rho=28.0,
        beta=8/3
    )
X = xyz.T  # (N, 3)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # 标准化

#parameters initialization
in_dim=out_dim=n_nodes=X.shape[1]
N_washout = 100
N_rep = 10
N_train=5000
N_test=1000
N_start=1000

#split dataset
def split_dataset(X, N_start, N_washout, N_train, N_test):
    X = X[N_start:, :]
    X_washout = X[:N_washout]
    X_train = X[N_washout:N_washout + N_train]
    Y_train = X[N_washout + 1:N_washout + N_train + 1]
    Y_test = X[N_washout + N_train: N_washout + N_train + N_test]
    return X_washout, X_train, Y_train, Y_test

def train_single_node(n_units, X_washout, X_train, Y_train, Y_test, N_rep, N_washout, N_evo, alpha, sigma_in, rho, option, tikh, use_norm):
    pirc = PIRC(n_units, in_dim, out_dim, sigma_in, rho, alpha, tikh, option)
    R=pirc.train(X_washout, X_train, Y_train)
    Trainning_Loss=pirc.trainning_loss( Y_train, R)
    if use_norm:
         CausalIndex = normalize(cal_CausalIndex_PIRC(pirc, Y_test, N_rep, N_washout, N_evo).reshape(-1, 1),axis=1)
    else:
        CausalIndex = cal_CausalIndex_PIRC(pirc, Y_test, N_rep, N_washout, N_evo)
    return CausalIndex, Trainning_Loss

def train_all_nodes(X, n_nodes, n_units, N_evo, alpha, sigma_in, rho, option, tikh, use_norm):
    Expand_nodes_num = n_nodes + math.comb(n_nodes - 1, 2)
    CausalIndex_matrix = np.zeros((n_nodes, Expand_nodes_num))
    Trainning_Loss_List = np.zeros(n_nodes)

    def train_for_position(position):
        X_hat = shift_column_to_first(X, position)
        X_washout, X_train, Y_train, Y_test = split_dataset(X_hat, N_start, N_washout, N_train, N_test)
        return train_single_node(n_units, X_washout, X_train, Y_train, Y_test, N_rep, N_washout, N_evo, alpha, sigma_in, rho, option, tikh, use_norm)

    results = Parallel(n_jobs=n_jobs)(
        delayed(train_for_position)(position) for position in range(n_nodes)
    )

    for position, (CausalIndex, Trainning_Loss) in enumerate(results):
        Trainning_Loss_List[position] = Trainning_Loss
        CausalIndex_matrix[position, :] = CausalIndex.ravel()

    return CausalIndex_matrix, Trainning_Loss_List

def objective(trial, X, n_nodes):
    N_evo = trial.suggest_categorical('N_evo', [10,20,30,40,50])
    N_units = trial.suggest_categorical('N_units',  [100,300,500,1000,2000])
    alpha = trial.suggest_float('alpha', 0.1,1, step=0.1)
    sigma_in = trial.suggest_categorical('sigma_in', [0.1,0.5,1.0])
    rho=trial.suggest_categorical('rho',  [0.7,0.8,0.9])
    tikh=trial.suggest_categorical('tikh', [1e-1,1e-2,1e-3,1e-4,1e-5])
    option = trial.suggest_categorical('option', [1])
    use_norm = trial.suggest_categorical('use_norm', [True])

    CausalIndex_matrix, Trainning_Loss_List = train_all_nodes(X, n_nodes, N_units,N_evo, alpha,
                                                sigma_in, rho, option, tikh, use_norm)
    Loss = np.mean(np.std(CausalIndex_matrix, axis=1))-np.mean(Trainning_Loss_List)
    return Loss

# 使用 Optuna 进行贝叶斯优化
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),  # 使用贝叶斯优化 (TPE) 采样器
    pruner=optuna.pruners.MedianPruner()   # 提前终止低效试验
)
study.optimize(lambda trial: objective(trial, X, n_nodes), n_trials=100)
# 输出最佳结果
print("\nBest Hyperparameters:", study.best_params)
print("Lowest Loss:", study.best_value)

# 获取最佳 TDI 矩阵
best_trial = study.best_trial
best_params = best_trial.params
best_N_evo = best_params['N_evo']
best_N_units = best_params['N_units']
best_alpha = best_params['alpha']
best_sigma_in=best_params['sigma_in']
best_rho=best_params['rho']
best_tikh=best_params['tikh']
best_option=best_params['option']
best_use_norm = best_params['use_norm']
CausalIndex_matrix, Trainning_Loss_List = train_all_nodes(X, n_nodes, best_N_units, best_N_evo, best_alpha, best_sigma_in, best_rho, best_option, best_tikh ,best_use_norm)
sns.heatmap(CausalIndex_matrix, cmap='gray', annot=True, fmt='.1f', cbar=True, xticklabels=5, yticklabels=True)
plt.show()
print("Trainning_Loss_List:", Trainning_Loss_List)






