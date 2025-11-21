import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
from Model.PIRC_torch import PIRC_Torch as PIRC
from Data_gen import generate_kuramoto_data

# %% parameters initialization
parser = argparse.ArgumentParser()
parser.add_argument('--node_id', type=int, default=0, help='Node ID (0-indexed)')
parser.add_argument('--GPU_id', type=int, default=0, help='GPU ID (0-indexed)')
parser.add_argument('--node_num', type=int, default=3, help='Number of nodes in the system')
parser.add_argument('--Pair_strength', type=float, default=0.1, help='Pairwise interaction strength')
parser.add_argument('--Tri_strength', type=float, default=0.1, help='Three-way interaction strength')
parser.add_argument('--Normalization', type=int, default=1, help='Whether to normalize the data')
args = parser.parse_args()

node_id = args.node_id
device = f"cuda:{args.GPU_id}"  # GPU id
in_dim= args.node_num
out_dim=3
# ExpandDim = math.comb(in_dim - 1, 2) * 2 + math.comb(in_dim, 1) * 2
ExpandNodes = in_dim + math.comb(in_dim - 1, 2)
N_washout = 100
# N_train=1000
N_test=100
N_start=1000
block_dim = 2  # 1 or 2
use_Sin = False
mode = 0
rep=10

# data_file = Path(f"data/NodeNum_{args.node_num}_PairStrength_{args.Pair_strength}_TriStrength_{args.Tri_strength}.pkl")
# if data_file.exists():
#     with open(data_file, "rb") as f:
#         a2, a3, data1 = pickle.load(f)

# %% 需要用的时候读取
param_file = f'Parameters/Test_node_{args.node_id}_PairStrength_{args.Pair_strength}_TriStrength_{args.Tri_strength}_out_dim_{out_dim}_UseSin_False_Normalization_1_block_dim2.pkl'
# param_file = 'Parameters/node_1_PairStrength_0.1_TriStrength_0.1_out_dim_3_UseSin_False_Normalization_1_block_dim2.pkl'
with open(param_file, 'rb') as f:
    # best_params, _, _, _= pickle.load(f)
    best_params, a2, a3, data = pickle.load(f)

# param_file = f'Parameters/node_{args.node_id}_PairStrength_{args.Pair_strength}_TriStrength_{args.Tri_strength}_out_dim_{out_dim}_UseSin_False_Normalization_1_block_dim2.pkl'
# # param_file = 'Parameters/node_1_PairStrength_0.1_TriStrength_0.1_out_dim_3_UseSin_False_Normalization_1_block_dim2.pkl'
# with open(param_file, 'rb') as f:
#     # best_params, _, _, _= pickle.load(f)
#     best_params, a2, a3, data = pickle.load(f)




# data_file = Path(f"data/NodeNum_{args.node_num}_PairStrength_{args.Pair_strength}_TriStrength_{args.Tri_strength}.pkl")
# if data_file.exists():
#     with open(data_file, "rb") as f:
#         a2, a3, data = pickle.load(f)
# param_file = f"Parameters/2025-11-18/N_train_10000_NodeNum_3_node_{args.node_id}_PairStrength_{args.Pair_strength}_TriStrength_{args.Tri_strength}_Normalization_1_block_dim2.pkl"
# with open(param_file, 'rb') as f:
#     # best_params, a2, a3, data = pickle.load(f)
#     best_params = pickle.load(f)

# with open('data/kuramoto_data.pkl', 'rb') as f:
#     a2, a3, data = pickle.load(f)
# _, _, data = generate_kuramoto_data(n=in_dim, dt=0.01, steps=10000, Pair_strength = args.Pair_strength, Tri_strength = args.Tri_strength, a2=a2, a3=a3)
X = torch.tensor(data, dtype=torch.float32, device=device)  # (N, 3)
# X = (X - X.mean(dim=0)) / X.std(dim=0) if args.Normalization == 1 else X  # 标准化
X = shift_column_to_first(X, node_id)

def trainLoss_single_node(in_dim, out_dim, n_units, X_washout, X_train, Y_train, Y_test, alpha, sigma_in, rho, option, tikh):
    Structured_W = generate_structured_w_v4(in_dim, ExpandNodes, device, block_dim = block_dim)
    Win, Wres = Partitioned_Win_Wres(n_units, ExpandNodes, device, block_dim = block_dim)
    pirc = PIRC(n_units, in_dim, out_dim, Structured_W, Win, Wres, sigma_in, rho, alpha, tikh, option, device, block_dim = block_dim, use_Sin=use_Sin, mode = mode)
    R, Training_Loss, X_predict=pirc.train(X_washout, X_train, Y_train)
    Y_test_predict, Test_Loss = pirc.Prediction(Y_test, R, N_test, Y_train)
    return R, Training_Loss, X_predict, Y_test_predict, Test_Loss

best_n_units = best_params['n_units']
best_alpha = best_params['alpha']
best_sigma_in=best_params['sigma_in']
best_rho=best_params['rho']
best_tikh=best_params['tikh']
best_option=best_params['option']
best_N_train=best_params['N_train']
best_out_dim=best_params['out_dim']
best_Normalization=best_params['Normalization']

# Structured_W = generate_structured_w(in_dim, ExpandNodes, device)
# Win, Wres = Partitioned_Win_Wres(best_n_units, ExpandNodes, device)

X = (X - X.mean(dim=0)) / X.std(dim=0) if best_Normalization == 1 else X  # 标准化
X_washout, X_train, Y_train, Y_test = split_dataset(X, N_start, N_washout, best_N_train, N_test)
List_Y_test_predict = []
for rep in range(rep):
    R, Training_Loss, Y_train_predict, Y_test_predict, Test_Loss= trainLoss_single_node(in_dim, best_out_dim, best_n_units, X_washout, X_train, Y_train, Y_test, best_alpha, best_sigma_in, best_rho, best_option, best_tikh)
    if Test_Loss[0]<1:
        List_Y_test_predict.append(Y_test_predict)
arr = np.stack(List_Y_test_predict, axis=0)
# 在 rep 维上求平均，结果形状为 (n_samples, out_dim) 或 (n_samples,)
mean_Y_test_predict = arr.mean(axis=0)
print("Training_Loss with best hyperparameters:", Training_Loss)
print("Test_Loss with best hyperparameters:", Test_Loss[0])

# %% Training phase visualization
fig, axes = plt.subplots(Y_train_predict.shape[1], 1, figsize=(10, 12))
if Y_train_predict.shape[1] == 1:
    axes = [axes]
for i in range(Y_train_predict.shape[1]):
    axes[i].plot(Y_train[:, i].cpu().numpy(), label='True')
    axes[i].plot(Y_train_predict[:, i], label='RC')
for ax in axes:
    ax.legend(loc='best')
plt.tight_layout()
os.makedirs('fig', exist_ok=True)
plt.savefig('fig/Validation_train_phase.png')
plt.show()

# %% Testing phase visualization
fig, axes = plt.subplots(ExpandNodes+1, 1, figsize=(10, 12))
for step in [10, 25, 50, 100]:
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

ax.set_ylabel('Value')
ax.set_xlabel('Time Steps (testing phase)')
ax.legend(loc='best')
plt.ylim([-2, 2])
plt.tight_layout()
os.makedirs('fig', exist_ok=True)
plt.savefig('fig/Validation_Test_phase.png')
plt.show()




