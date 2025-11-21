import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
from Model.PIRC_torch import PIRC_Torch as PIRC
# 如果要换模型验证，一共有两个地方要改： 1. 上面这一行 2. out_dim

# %% parameters initialization
device= "cuda:1"  # GPU id
in_dim= 3
out_dim=3
ExpandDim = math.comb(in_dim - 1, 2) * 2 + math.comb(in_dim, 1) * 2
ExpandNodes = in_dim + math.comb(in_dim - 1, 2)
N_washout = 100
N_train=5000
N_test=200
N_start=1000
position = int(device.split(":")[1])  # 0,1,2 分别对应 X,Y,Z 三个变量
param_file = f'Parameters/best_params_{"x" if position == 0 else "y" if position == 1 else "z"}.pkl'
mode = 0
rep=10

# %% 需要用的时候读取
with open('data/rossler_data.pkl', 'rb') as f:
    t, states = pickle.load(f)
X = torch.tensor(states, dtype=torch.float32, device=device)  # (N, 3)
X = (X - X.mean(dim=0)) / X.std(dim=0)  # 标准化
X = shift_column_to_first(X, position)
X_washout, X_train, Y_train, Y_test = split_dataset(X, N_start, N_washout, N_train, N_test, in_dim=in_dim, out_dim=out_dim)

def trainLoss_single_node(n_units, X_washout, X_train, Y_train,  alpha, sigma_in, rho, option, tikh):
    Structured_W = generate_structured_w_v1(in_dim, ExpandNodes, device)
    Win, Wres = Partitioned_Win_Wres(n_units, ExpandNodes, device)
    pirc = PIRC(n_units, in_dim, out_dim, Structured_W, Win, Wres, sigma_in, rho, alpha, tikh, option, device, mode = mode)
    R, Training_Loss, X_predict=pirc.train(X_washout, X_train, Y_train)
    Y_test_predict, Test_Loss = pirc.Prediction( Y_test, R, N_test, Y_train)
    return R, Training_Loss, X_predict, Y_test_predict, Test_Loss

with open(param_file, 'rb') as f:
    best_params = pickle.load(f)

best_n_units = best_params['n_units']
best_alpha = best_params['alpha']
best_sigma_in=best_params['sigma_in']
best_rho=best_params['rho']
best_tikh=best_params['tikh']
best_option=best_params['option']

# Structured_W = generate_structured_w_v1(in_dim, ExpandNodes, device)
# Win, Wres = Partitioned_Win_Wres(best_n_units, ExpandNodes, device)
List_Y_test_predict = []
for rep in range(rep):
    R, Training_Loss, Y_train_predict, Y_test_predict, Test_Loss= trainLoss_single_node(best_n_units, X_washout, X_train, Y_train, best_alpha, best_sigma_in, best_rho, best_option, best_tikh)
    if Test_Loss[0]<1:
        List_Y_test_predict.append(Y_test_predict)
arr = np.stack(List_Y_test_predict, axis=0)
# 在 rep 维上求平均，结果形状为 (n_samples, out_dim) 或 (n_samples,)
mean_Y_test_predict = arr.mean(axis=0)
print("Training_Loss with best hyperparameters:", Training_Loss)
print("Test_Loss with best hyperparameters:", Test_Loss[0])

# %% Training phase visualization
fig, axes = plt.subplots(Y_train_predict.shape[1], 1, figsize=(10, 12))
for i in range(Y_train_predict.shape[1]):
    axes[i].plot(Y_train[-500:, i].cpu().numpy(), label='True')
    axes[i].plot(Y_train_predict[-500:, i], label='RC')
for ax in axes:
    ax.legend(loc='best')
plt.tight_layout()
os.makedirs('fig', exist_ok=True)
plt.savefig('fig/Validation_train_phase.png')
plt.show()

# %% Testing phase visualization
fig, axes = plt.subplots(ExpandNodes+1, 1, figsize=(10, 12))
for step in [10, 25, 50]:
    for i in range(ExpandNodes+1):
        axes[i].plot(Y_test[:step, 0].cpu().numpy(), label='True')
        axes[i].plot(mean_Y_test_predict[i][:step, 0], label='RC')
        mse = ((mean_Y_test_predict[i][:step, 0] - Y_test[:step, 0].cpu().numpy()) ** 2).mean()
        print(f"第{i}个节点在{step}步的MSE: {mse}")
    print('-----------------------------------')
axes[0].set_ylabel('without intervention')
for i in range(1,ExpandNodes):
    axes[i].set_ylabel(f'intervention X{i}')
axes[ExpandNodes].set_xlabel('Time Steps (testing phase)')
for ax in axes:
    ax.legend(loc='best')
plt.tight_layout()
plt.savefig('fig/Validation_Test_phase.png')
plt.show()

# %% Testing phase visualization (all curves on one plot)
fig, ax = plt.subplots(figsize=(10, 6))
step = N_test
true_series = Y_test[:step, 0].cpu().numpy()
ax.plot(true_series, label='True', color='k', linewidth=2)

for i in range(ExpandNodes + 1):
    pred = mean_Y_test_predict[i][:step, 0]
    ax.plot(pred, label=f'RC intervention {i}', alpha=0.8)
    mse = ((pred - true_series) ** 2).mean()
    print(f"第{i}个节点在{step}步的MSE: {mse}")

ax.set_ylabel('Value')
ax.set_xlabel('Time Steps (testing phase)')
ax.legend(loc='best')
plt.ylim([-2, 2])
plt.tight_layout()
os.makedirs('fig', exist_ok=True)
plt.savefig('fig/Validation_Test_phase.png')
plt.show()






