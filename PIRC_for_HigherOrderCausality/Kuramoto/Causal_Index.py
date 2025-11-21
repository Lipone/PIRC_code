import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
from Model.PIRC_torch import PIRC_Torch as PIRC
from Data_gen import generate_kuramoto_data

def trainLoss_single_node(in_dim, out_dim, n_units, X_washout, X_train, Y_train, Y_test, alpha, sigma_in, rho, option, tikh):
    Structured_W = generate_structured_w_v4(in_dim, ExpandNodes, device, block_dim = block_dim)
    Win, Wres = Partitioned_Win_Wres(n_units, ExpandNodes, device, block_dim = block_dim)
    pirc = PIRC(n_units, in_dim, out_dim, Structured_W, Win, Wres, sigma_in, rho, alpha, tikh, option, device, block_dim = block_dim, use_Sin=use_Sin, mode = mode)
    R, Training_Loss, X_predict=pirc.train(X_washout, X_train, Y_train)
    Y_test_predict, Test_Loss = pirc.Prediction(Y_test, R, N_test, Y_train)
    return R, Training_Loss, X_predict, Y_test_predict, Test_Loss

# %% parameters initialization
parser = argparse.ArgumentParser()
parser.add_argument('--GPU_id', type=int, default=0, help='GPU ID (0-indexed)')
parser.add_argument('--node_num', type=int, default=3, help='Number of nodes in the system')
parser.add_argument('--Pair_strength', type=float, default=0.1, help='Pairwise interaction strength')
parser.add_argument('--Tri_strength', type=float, default=0.1, help='Three-way interaction strength')
parser.add_argument('--Normalization', type=int, default=1, help='Whether to normalize the data')
args = parser.parse_args()

device = f"cuda:{args.GPU_id}"  # GPU id
in_dim= args.node_num
out_dim=3
ExpandNodes = in_dim + math.comb(in_dim - 1, 2)
N_washout = 100
N_test=200
N_start=1000
block_dim = 2  # 1 or 2
use_Sin = False
mode = 0
rep=10
# step=50
Error1 = np.zeros((args.node_num, ExpandNodes))
Error2 = np.zeros((args.node_num, ExpandNodes))

data_file = Path(f"data/NodeNum_{args.node_num}_PairStrength_{args.Pair_strength}_TriStrength_{args.Tri_strength}.pkl")
if data_file.exists():
    with open(data_file, "rb") as f:
        a2, a3, data1 = pickle.load(f)
#
# print("a2:\n",a2)
# print("a3:\n",a3)

# param_file = f'Parameters/node_0_PairStrength_{args.Pair_strength}_TriStrength_{args.Tri_strength}_out_dim_{out_dim}_UseSin_False_Normalization_1_block_dim2.pkl'
# with open(param_file, 'rb') as f:
#     _, _, _, data2 = pickle.load(f)
#     # _, a2, a3, data = pickle.load(f)

data=data1

for step in [10, 25, 50, 75, 100]:
    print(f"--------------------{step}步--------------------")
    for node_id in range(args.node_num):
        param_file = f'Parameters/Test_node_{node_id}_PairStrength_{args.Pair_strength}_TriStrength_{args.Tri_strength}_out_dim_{out_dim}_UseSin_False_Normalization_1_block_dim2.pkl'
        with open(param_file, 'rb') as f:
            best_params1, a2, a3, _ = pickle.load(f)
            # _, a2, a3, data = pickle.load(f)

        param_file = f"Parameters/2025-11-19/N_train_20000_N_test_25_NodeNum_3_node_{node_id}_PairStrength_{args.Pair_strength}_TriStrength_{args.Tri_strength}_Normalization_1_block_dim2.pkl"
        with open(param_file, 'rb') as f:
            # best_params, a2, a3, data = pickle.load(f)
            best_params2 = pickle.load(f)

        best_params = best_params2

        X = torch.tensor(data, dtype=torch.float32, device=device)  # (N, 3)
        X = shift_column_to_first(X, node_id)

        best_n_units = best_params['n_units']
        best_alpha = best_params['alpha']
        best_sigma_in=best_params['sigma_in']
        best_rho=best_params['rho']
        best_tikh=best_params['tikh']
        best_option=best_params['option']
        best_N_train= best_params['N_train']
        best_out_dim= best_params['out_dim']
        best_Normalization=best_params['Normalization']

        X = (X - X.mean(dim=0)) / X.std(dim=0) if best_Normalization == 1 else X  # 标准化
        X_washout, X_train, Y_train, Y_test = split_dataset(X, N_start, N_washout, best_N_train, N_test)
        List_Y_test_predict = []
        for rep in range(rep):
            R, Training_Loss, Y_train_predict, Y_test_predict, Test_Loss= trainLoss_single_node(in_dim, best_out_dim, best_n_units, X_washout, X_train, Y_train, Y_test, best_alpha, best_sigma_in, best_rho, best_option, best_tikh)
            # if Test_Loss[0]<1:
            #     List_Y_test_predict.append(Y_test_predict)
            List_Y_test_predict.append(Y_test_predict)
        arr = np.stack(List_Y_test_predict, axis=0)
        mean_Y_test_predict = arr.mean(axis=0)

        mse=((mean_Y_test_predict[0][:step, 0] - Y_test[:step, 0].cpu().numpy()) ** 2).mean()
        print(f"Mean squared error for node {node_id} in {step} steps: ", mse)

        for i in range(1, ExpandNodes + 1):
            mse1 = ((mean_Y_test_predict[i][:step, 0] - mean_Y_test_predict[0][:step, 0]) ** 2).mean()
            Error1[node_id, i-1] = mse1
            mse2 = ((mean_Y_test_predict[i][:step, 0] - Y_test[:step, 0].cpu().numpy()) ** 2).mean()
            Error2[node_id, i - 1] = mse2

    # print("\nError1 矩阵:")
    # print(Error1)
    #
    # print("\nError2 矩阵:")
    # print(Error2)

    Error1 = shift_column_for_Causal_Matrix(Error1)
    Error2 = shift_column_for_Causal_Matrix(Error2)
    np.fill_diagonal(Error1, 1.0)
    np.fill_diagonal(Error2, 1.0)

    # print("\nshift_Error1 矩阵:")
    # print(Error1)

    # print("\nshift_Error2 矩阵:")
    print(Error2)


# import matplotlib.pyplot as plt
# import seaborn as sns
#
# rows = args.node_num
# cols = ExpandNodes
# data = Error1  # 使用处理后的 Error1 矩阵
# for i in range(min(rows, data.shape[1])):
#     data[i, i] = np.nan  # 对角线置为 NaN 以显示为白色
#
# cmap = plt.get_cmap('GnBu')
# cmap.set_bad(color='white')  # NaN 显示为白色
#
# plt.figure(figsize=(8, 6))
# ax = sns.heatmap(data, cmap=cmap, cbar_kws={'label': 'MSE'})
# ax.plot([0, rows], [0, rows], linestyle='--', color='gray', linewidth=1)  # 灰色虚线
# plt.title('Error Heatmap')
# plt.xlabel('Expanded node index (excluding reference 0)')
# plt.ylabel('Source node id')
# plt.tight_layout()
# plt.savefig(f'fig/Error_heatmap_Pair_{args.Pair_strength}_Tri_{args.Tri_strength}.png', dpi=300)
# plt.show()




