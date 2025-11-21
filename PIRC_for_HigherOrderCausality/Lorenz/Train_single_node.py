import optuna
import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Lorenz.Data_gen import *
from Model.PIRC import PIRC
from Model.Causal_Index import cal_CausalIndex_PIRC
from Library import *
import pickle
# rng = np.random.default_rng(42)
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed

# %% 需要用的时候读取
with open('data/lorenz_data.pkl', 'rb') as f:
    t, xyz = pickle.load(f)
X = xyz.T  # (N, 3)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # 标准化
print("nomlized data:", X)

# %% parameters initialization
position=1  # 0,1,2 分别对应 X,Y,Z 三个变量
param_file = f'Parameters/best_params_{"x" if position == 0 else "y" if position == 1 else "z"}.pkl'
in_dim=out_dim=X.shape[1]
N_washout = 100
N_rep = 10
N_train=5000
N_test=100
N_start=1000
n_trials=200

#split dataset
def split_dataset(X, N_start, N_washout, N_train, N_test):
    X = X[N_start:, :]
    X_washout = X[:N_washout]
    X_train = X[N_washout:N_washout + N_train]
    Y_train = X[N_washout + 1:N_washout + N_train + 1]
    Y_test = X[N_washout + N_train: N_washout + N_train + N_test]
    return X_washout, X_train, Y_train, Y_test
X = shift_column_to_first(X, position)
X_washout, X_train, Y_train, Y_test = split_dataset(X, N_start, N_washout, N_train, N_test)

def trainLoss_single_node(n_units, X_washout, X_train, Y_train,  alpha, sigma_in, rho, option, tikh):
    pirc = PIRC(n_units, in_dim, out_dim, sigma_in, rho, alpha, tikh, option)
    R, Trainning_Loss, X_predict=pirc.train(X_washout, X_train, Y_train)
    Y_test_predict, Test_Loss = pirc.Prediction(Y_test, R, N_test)
    return R, Trainning_Loss, X_predict, Y_test_predict, Test_Loss

def objective(trial, X):
    # N_evo = trial.suggest_categorical('N_evo', [10,20,30,40,50])
    N_units = trial.suggest_categorical('N_units',  [1000,2000,3000,4000,5000])
    alpha = trial.suggest_float('alpha', 0.1,1, step=0.1)
    sigma_in = trial.suggest_float('sigma_in', 0.1,1, step=0.1)
    rho=trial.suggest_float('rho',  0.1,1, step=0.1)
    tikh=trial.suggest_categorical('tikh', [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8])
    option = trial.suggest_categorical('option', [1,2,3])
    # use_norm = trial.suggest_categorical('use_norm', [True])
    R, Trainning_Loss, Y_train_Predict, _, Test_Loss = trainLoss_single_node(N_units, X_washout, X_train, Y_train,  alpha, sigma_in, rho, option, tikh)
    return Test_Loss[0]

# %% train
# 使用 Optuna 进行贝叶斯优化
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),  # 使用贝叶斯优化 (TPE) 采样器
    pruner=optuna.pruners.MedianPruner()   # 提前终止低效试验
)
study.optimize(lambda trial: objective(trial, X), n_trials=n_trials)
# 输出最佳结果
print("\nBest Hyperparameters:", study.best_params)
print("Lowest Loss:", study.best_value)

best_params = study.best_params
with open(param_file, 'wb') as f:
    pickle.dump(best_params, f)

# 获取最佳 TDI 矩阵
best_trial = study.best_trial
best_params = best_trial.params
best_N_units = best_params['N_units']
best_alpha = best_params['alpha']
best_sigma_in=best_params['sigma_in']
best_rho=best_params['rho']
best_tikh=best_params['tikh']
best_option=best_params['option']

R, Trainning_Loss, Y_train_Predict, _, Test_Loss= trainLoss_single_node(best_N_units, X_washout, X_train, Y_train, best_alpha, best_sigma_in, best_rho, best_option, best_tikh)
print("\nBest Hyperparameters:", study.best_params)
print("Trainning_Loss with best hyperparameters:", Trainning_Loss)
print("Test_Loss with best hyperparameters:", Test_Loss[0])

# %% test
with open(param_file, 'rb') as f:
    best_params = pickle.load(f)
pirc = PIRC(best_N_units, in_dim, out_dim, best_sigma_in, best_rho, best_alpha, best_tikh, best_option)
R, Trainning_Loss, Y_train_predict = pirc.train(X_washout, X_train, Y_train)
Y_test_predict, Test_Loss = pirc.Prediction(Y_test, R, N_test)
print("Training Loss:", Trainning_Loss)
print("Test_Loss:", Test_Loss)

# %% Training phase visualization
fig, axes = plt.subplots(Y_train_predict.shape[1], 1, figsize=(10, 12))
for i in range(Y_train_predict.shape[1]):
    axes[i].plot(X_train[-100:, i], label='True')
    axes[i].plot(Y_train_predict[-100:, i], label='RC')
axes[0].set_ylabel('X')
axes[1].set_ylabel('Y')
axes[2].set_ylabel('Z')
axes[2].set_xlabel('Time Steps (training phase)')
for ax in axes:
    ax.legend(loc='best')
plt.tight_layout()
plt.savefig('fig/train_phase.png')
plt.show()

# %% Testing phase visualization
fig, axes = plt.subplots(pirc.ExpandNodes+1, 1, figsize=(10, 12))
for i in range(pirc.ExpandNodes+1):
    axes[i].plot(Y_test[:50, 0], label='True')
    axes[i].plot(Y_test_predict[i][:50, 0], label='RC')
axes[0].set_ylabel('without intervention')
for i in range(1,pirc.ExpandNodes):
    axes[i].set_ylabel(f'intervention X{i}')
axes[pirc.ExpandNodes].set_xlabel('Time Steps (testing phase)')
for ax in axes:
    ax.legend(loc='best')
plt.tight_layout()
plt.savefig('fig/Test_phase.png')
plt.show()






