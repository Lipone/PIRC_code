import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
from Model.PIRC_torch import PIRC_Torch as PIRC

# %% parameters initialization
device= "cuda:1"  # GPU id
in_dim= 3
out_dim=3
ExpandDim = math.comb(in_dim - 1, 2) * 2 + math.comb(in_dim, 1) * 2
ExpandNodes = in_dim + math.comb(in_dim - 1, 2)
N_washout = 100
N_rep = 10
N_train=5000
N_test=25
N_start=1000
n_trials=100

position = int(device.split(":")[1])  # 0,1,2 分别对应 X,Y,Z 三个变量
param_file = f'Parameters/best_params_{"x" if position == 0 else "y" if position == 1 else "z"}.pkl'

# %% 需要用的时候读取
with open('data/rossler_data.pkl', 'rb') as f:
    t, states = pickle.load(f)
X = torch.tensor(states, dtype=torch.float32, device=device)  # (N, 3)
X = shift_column_to_first(X, position)
X = (X - X.mean(dim=0)) / X.std(dim=0)  # 标准化
X_washout, X_train, Y_train, Y_test = split_dataset(X, N_start, N_washout, N_train, N_test)

def trainLoss_single_node(n_units, X_washout, X_train, Y_train,  alpha, sigma_in, rho, option, tikh):
    Structured_W = generate_structured_w(in_dim, ExpandDim, device)
    Win, Wres = Partitioned_Win_Wres(n_units, ExpandNodes, device)
    pirc = PIRC(n_units, in_dim, out_dim, Structured_W, Win, Wres, sigma_in, rho, alpha, tikh, option, device)
    R, Training_Loss, X_predict=pirc.train(X_washout, X_train, Y_train)
    Y_test_predict, Test_Loss = pirc.Prediction(Y_test, R, N_test)
    return R, Training_Loss, X_predict, Y_test_predict, Test_Loss

def objective(trial, X):
    n_units = trial.suggest_categorical('n_units',  [1000,2000,3000,4000,5000])
    alpha = trial.suggest_float('alpha', 0.1,1, step=0.1)
    sigma_in = trial.suggest_float('sigma_in', 0.1,1, step=0.1)
    rho=trial.suggest_float('rho',  0.1,1, step=0.1)
    tikh=trial.suggest_categorical('tikh', [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8])
    option = trial.suggest_categorical('option', [1,2,3])
    # trial.set_user_attr("Structured_W", Structured_W.tolist())
    # trial.set_user_attr("Win", Win.tolist())
    # trial.set_user_attr("Wres", Wres.tolist())
    test_losses = []
    for round in range(N_rep):
        R, Training_Loss, Y_train_predict, _, Test_Loss = trainLoss_single_node(n_units, X_washout, X_train, Y_train,  alpha, sigma_in, rho, option, tikh)
        test_losses.append(Test_Loss[0])  # 保存每次的 Test_Loss[0]
    # print("Test Loss for each rep: {}".format(test_losses))
    return sum(test_losses) / len(test_losses)

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
best_n_units = best_params['n_units']
best_alpha = best_params['alpha']
best_sigma_in=best_params['sigma_in']
best_rho=best_params['rho']
best_tikh=best_params['tikh']
best_option=best_params['option']

# %% test
Structured_W = generate_structured_w(in_dim, ExpandDim, device)
Win, Wres = Partitioned_Win_Wres(best_n_units, ExpandNodes, device)
R, Training_Loss, Y_train_predict, Y_test_predict, Test_Loss= trainLoss_single_node(best_n_units, X_washout, X_train, Y_train, best_alpha, best_sigma_in, best_rho, best_option, best_tikh)
print("\nBest Hyperparameters:", study.best_params)
print("Training_Loss with best hyperparameters:", Training_Loss)
print("Test_Loss with best hyperparameters:", Test_Loss[0])
# with open('best_params.pkl', 'rb') as f:
#     best_params = pickle.load(f)
# pirc = PIRC(best_n_units, in_dim, out_dim, best_sigma_in, best_rho, best_alpha, best_tikh, best_option, Structured_W, Win, Wres)
# R, Training_Loss, Y_train_predict = pirc.train(X_washout, X_train, Y_train)
# Y_test_predict, Test_Loss = pirc.Prediction(Y_test, R, N_test)
# print("Training Loss:", Training_Loss)
# print("Test_Loss:", Test_Loss)

# %% Training phase visualization
fig, axes = plt.subplots(Y_train_predict.shape[1], 1, figsize=(10, 12))
for i in range(Y_train_predict.shape[1]):
    axes[i].plot(Y_train[-500:, i].cpu().numpy(), label='True')
    axes[i].plot(Y_train_predict[-500:, i], label='RC')
axes[0].set_ylabel('X')
axes[1].set_ylabel('Y')
axes[2].set_ylabel('Z')
axes[2].set_xlabel('Time Steps (training phase)')
for ax in axes:
    ax.legend(loc='best')
plt.tight_layout()
os.makedirs('fig', exist_ok=True)
plt.savefig('fig/train_phase.png')
plt.show()

# %% Testing phase visualization
fig, axes = plt.subplots(ExpandNodes+1, 1, figsize=(10, 12))
for i in range(ExpandNodes+1):
    axes[i].plot(Y_test[:, 0].cpu().numpy(), label='True')
    axes[i].plot(Y_test_predict[i][:, 0], label='RC')
axes[0].set_ylabel('without intervention')
for i in range(1,ExpandNodes):
    axes[i].set_ylabel(f'intervention X{i}')
axes[ExpandNodes].set_xlabel('Time Steps (testing phase)')
for ax in axes:
    ax.legend(loc='best')
plt.tight_layout()
plt.savefig('fig/Test_phase.png')
plt.show()






