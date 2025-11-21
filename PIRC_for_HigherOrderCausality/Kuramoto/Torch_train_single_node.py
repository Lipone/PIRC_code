import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
from Model.PIRC_torch import PIRC_Torch as PIRC
from Data_gen import generate_kuramoto_data

# %% parameters initializationN
parser = argparse.ArgumentParser()
parser.add_argument('--node_id', type=int, default=0, help='Node ID (0-indexed)')
parser.add_argument('--GPU_id', type=int, default=0, help='GPU ID (0-indexed)')
parser.add_argument('--node_num', type=int, default=3, help='Number of nodes in the system')
parser.add_argument('--Pair_strength', type=float, default=0.1, help='Pairwise interaction strength')
parser.add_argument('--Tri_strength', type=float, default=0.1, help='Three-way interaction strength')
parser.add_argument('--block_dim', type=int, default=2, help='Block dimension for structured W')
parser.add_argument('--use_Sin', action='store_true', help='Use Sin nonlinearity (flag, default False)')
parser.add_argument('--N_test', type=int, default=25, help='Number of test steps')
# parser.add_argument('--Normalization', type=int, default=1, help='Whether to normalize the data')
args = parser.parse_args()

node_id = args.node_id
device = f"cuda:{args.GPU_id}"  # GPU id
in_dim= args.node_num
block_dim = args.block_dim  # 1 or 2
use_Sin = args.use_Sin
ExpandNodes = in_dim + math.comb(in_dim - 1, 2)
N_washout = 100
N_rep = 10
# N_train=5000
N_test=100
N_start=1000
n_trials=500
os.makedirs('Parameters', exist_ok=True)
# THRESHOLD = 1e-4  # 当 best_value < THRESHOLD 时提前停止

# %% Data generation
# with open('data/rossler_data.pkl', 'rb') as f:
#     t, states = pickle.load(f)

a2 = [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.1],
        [0.1, 0.0, 0.0]
    ]

a3 = [
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1],
        [0.0, 0.0, 0.0]
    ],
    [
        [0.0, 0.0, 0.1],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ],
    [
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]
]

a2, a3, data = generate_kuramoto_data(n=in_dim, dt=0.01, steps=100000, Pair_strength = args.Pair_strength, Tri_strength = args.Tri_strength, a2=a2, a3=a3)
# print("a2:\n", a2)
# print("a3:\n", a3)
X = torch.tensor(data, dtype=torch.float32, device=device)  # (N, 3)
X = shift_column_to_first(X, node_id)
# X_washout, X_train, Y_train, Y_test = split_dataset(X, N_start, N_washout, N_train, N_test)

# %% hyperparameter optimization with optuna
def trainLoss_single_node(in_dim, out_dim, n_units, X_washout, X_train, Y_train, Y_test, alpha, sigma_in, rho, option, tikh):
    Structured_W = generate_structured_w_v4(in_dim, ExpandNodes, device, block_dim = block_dim)
    Win, Wres = Partitioned_Win_Wres(n_units, ExpandNodes, device, block_dim = block_dim)
    pirc = PIRC(n_units, in_dim, out_dim, Structured_W, Win, Wres, sigma_in, rho, alpha, tikh, option, device, block_dim = block_dim, use_Sin=use_Sin)
    R, Training_Loss, X_predict=pirc.train(X_washout, X_train, Y_train)
    Y_test_predict, Test_Loss = pirc.Prediction(Y_test, R, N_test, Y_train)
    return R, Training_Loss, X_predict, Y_test_predict, Test_Loss

def objective(trial, X):
    n_units = trial.suggest_int('n_units', 1000, 3000, step=1000)
    alpha = trial.suggest_float('alpha', 0.1,1, step=0.1)
    sigma_in = trial.suggest_float('sigma_in', 0.1,1, step=0.1)
    rho=trial.suggest_float('rho',  0.1,1, step=0.1)
    tikh=trial.suggest_categorical('tikh', [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10])
    option = trial.suggest_categorical('option', [2,3,4,5])
    # trial.set_user_attr("Structured_W", Structured_W.tolist())
    # trial.set_user_attr("Win", Win.tolist())
    # trial.set_user_attr("Wres", Wres.tolist())
    # N_train =  trial.suggest_int('N_train', 5000, 50000, step=5000)
    N_train = trial.suggest_categorical('N_train', [20000])
    out_dim = trial.suggest_categorical('out_dim', [3])
    Normalization = trial.suggest_categorical('Normalization', [1])

    X = (X - X.mean(dim=0)) / X.std(dim=0) if Normalization == 1 else X  # 标准化
    X_washout, X_train, Y_train, Y_test = split_dataset(X, N_start, N_washout, N_train, N_test)
    test_losses = []
    for round in range(N_rep):
        R, Training_Loss, Y_train_predict, _, Test_Loss = trainLoss_single_node(in_dim, out_dim, n_units, X_washout, X_train, Y_train, Y_test, alpha, sigma_in, rho, option, tikh)
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
# study.optimize(lambda trial: objective(trial, X), n_trials=n_trials)
study.optimize(lambda trial: objective(trial, X), n_trials=n_trials, callbacks=[stop_when_low_enough])

# 输出最佳结果
print("Study finished.")
print("\nBest Hyperparameters:", study.best_params)
print("Lowest Loss:", study.best_value)


# 获取最佳参数
best_trial = study.best_trial
best_params = best_trial.params
best_n_units = best_params['n_units']
best_alpha = best_params['alpha']
best_sigma_in=best_params['sigma_in']
best_rho=best_params['rho']
best_tikh=best_params['tikh']
best_option=best_params['option']
best_N_train=best_params['N_train']
best_out_dim=best_params['out_dim']
best_Normalization=best_params['Normalization']

best_params = study.best_params
param_file = f'Parameters/Test_node_{node_id}_PairStrength_{args.Pair_strength}_TriStrength_{args.Tri_strength}_out_dim_{best_out_dim}_UseSin_{use_Sin}_Normalization_{best_Normalization}_block_dim{block_dim}.pkl'
with open(param_file, 'wb') as f:
    pickle.dump((best_params, a2, a3, data), f)

X = (X - X.mean(dim=0)) / X.std(dim=0) if best_Normalization == 1 else X  # 标准化
X_washout, X_train, Y_train, Y_test = split_dataset(X, N_start, N_washout, best_N_train, N_test)

# %% test
# Structured_W = generate_structured_w(in_dim, ExpandDim, device)
# Win, Wres = Partitioned_Win_Wres(best_n_units, ExpandNodes, device)
R, Training_Loss, Y_train_predict, Y_test_predict, Test_Loss= trainLoss_single_node(in_dim, best_out_dim, best_n_units, X_washout, X_train, Y_train, Y_test, best_alpha, best_sigma_in, best_rho, best_option, best_tikh)
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
if Y_train_predict.shape[1] == 1:
    axes = [axes]
for i in range(Y_train_predict.shape[1]):
    axes[i].plot(Y_train[-500:, i].cpu().numpy(), label='True')
    axes[i].plot(Y_train_predict[-500:, i], label='RC')
    axes[i].set_ylabel(f'node{i}')
axes[-1].set_xlabel('Time Steps (training phase)')
for ax in axes:
    ax.legend(loc='best')
plt.tight_layout()
os.makedirs('fig', exist_ok=True)
plt.savefig('fig/train.png')
# plt.show()

# %% Testing phase visualization
fig, axes = plt.subplots(ExpandNodes+1, 1, figsize=(10, 12))
for step in [25]:
    for i in range(ExpandNodes+1):
        axes[i].plot(Y_test[:, 0].cpu().numpy(), label='True')
        axes[i].plot(Y_test_predict[i][:, 0], label='RC')
        mse = ((Y_test_predict[i][:step, 0] - Y_test[:step, 0].cpu().numpy()) ** 2).mean()
        print(f"第{i}个节点在{step}步的MSE: {mse}")
axes[0].set_ylabel('without intervention')
for i in range(1,ExpandNodes):
    axes[i].set_ylabel(f'intervention Node{i}')
axes[ExpandNodes].set_xlabel('Time Steps (testing phase)')
for ax in axes:
    ax.legend(loc='best')
plt.tight_layout()
plt.savefig(f'fig/Test_node{node_id}_PairStrength_{args.Pair_strength}_TriStrength_{args.Tri_strength}_Normalization_{best_Normalization}_outdim_{best_out_dim}_UseSin_{use_Sin}_block_dim_{block_dim}.png')
# plt.show()






