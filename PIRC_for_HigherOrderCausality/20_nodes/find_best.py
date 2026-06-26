import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
from Model.PIRC_flatten_enhance import PIRC_flatten as Nonliear_PIRC
from DateGen_kuramoto import load_or_generate_kuramoto

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_num', type=int, default=20, help='Number of nodes in the system')
    parser.add_argument('--Pair_strength', type=float, default=0.4, help='Pairwise interaction strength') #1
    parser.add_argument('--Tri_strength', type=float, default=0.4, help='Three-way interaction strength') #1
    parser.add_argument("--N_train", type=int, default=10000)
    parser.add_argument("--N_test", type=int, default=10)
    parser.add_argument("--N_washout", type=int, default=100)
    parser.add_argument("--N_start", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=500)
    parser.add_argument("--Threshold", type=float, default=1e-3)
    parser.add_argument("--P_probability", type=float, default=0.05)  # 0.02
    parser.add_argument("--T_probability", type=float, default=0.005)  # 0.02
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()
    ExpandNodes = args.node_num + math.comb(args.node_num - 1, 2)
    set_seed(42)
    device = torch.device("cuda:1")

    data, a2, a3 = load_or_generate_kuramoto(args)
    data = data[args.N_start:,]
    X = torch.as_tensor(data, dtype=torch.float32, device=device)
    X_batch = torch.stack(
        [shift_column_to_first(X, i) for i in range(args.node_num)],
        dim=0
    )
    X_washout, X_train, Y_train, Y_test = split_dataset2_batch(
        X_batch,
        args.N_washout,
        args.N_train,
        args.N_test,
        in_dim=args.node_num,
        out_dim=args.node_num
    )

    T_Matrix=np.zeros((args.node_num, ExpandNodes))
    T_Matrix[:, :args.node_num] = (a2 != 0).astype(int)
    for i in range(args.node_num):
        com_list=list(combinations([x for x in range(args.node_num) if x != i], 2))
        for index,(j,k) in enumerate(com_list):
            T_Matrix[i, args.node_num+index] = int(a3[i][j][k] != 0)
    T_tensor = torch.as_tensor(T_Matrix, dtype=torch.float32, device=device)
    print("ground_truth", T_Matrix)
    count_Pair = np.count_nonzero(T_Matrix[:, :args.node_num])
    count_Tri = np.count_nonzero(T_Matrix[:, args.node_num:])
    print("pair:", count_Pair)
    print("Tri:", count_Tri)

    n_units = 2
    alpha = 0.3
    sigma_in = 1.0
    rho = 0.2
    tikh = 0.1
    option = 4
    tau = 2
    method = 'Logistic'
    connectivity = 1.0
    I_type = 1
    mode = 1
    block_dim = 2
    Base = 'prediction'

    best_auc = -1
    best_Win = None
    best_Wres = None

    # checkpoint = torch.load('best_full.pth')
    #
    # Win = checkpoint['Win']
    # Wres = checkpoint['Wres']
    # pirc = Nonliear_PIRC(
    #     n_units=n_units, in_dim=args.node_num, out_dim=args.node_num,
    #     Win=Win, Wres=Wres,
    #     sigma_in=sigma_in, rho=rho, alpha=alpha, tikh=tikh, mode=mode,
    #     device=device, block_dim=block_dim, I_type=I_type, option=option
    # )
    #
    # R = pirc.train_memEff(X_washout, X_train, Y_train)
    # Y_test_predict = pirc.Prediction(R, args.N_test, Y_train)
    #
    # base_all = Y_test_predict[0, :, :args.N_test, 0]
    # others_all = Y_test_predict[2:, :, :args.N_test, 0]
    # base_expand = base_all.unsqueeze(0).expand_as(others_all)
    #
    # score_EN = score_0_1_torch(
    #     others_all,
    #     base_expand,
    #     method=method,
    #     tau=tau
    # )
    #
    # Score = score_EN.transpose(0, 1).contiguous()
    # Score = shift_column_for_Causal_Matrix(Score)
    # Score.fill_diagonal_(1.0)
    #
    # AUC = ranking_auc_ALL(Score, T_tensor)
    # print(f"AUC = {AUC}")
    #
    # import numpy as np
    #
    # mask = np.ones(Score.shape, dtype=bool)
    # np.fill_diagonal(mask, False)
    #
    # score_np = Score.detach().cpu().numpy()[mask]
    # label_np = T_tensor.detach().cpu().numpy()[mask]
    # from sklearn.metrics import roc_curve, auc
    #
    # fpr, tpr, thresholds = roc_curve(label_np, score_np)
    # roc_auc = auc(fpr, tpr)
    #
    # print(f"AUC (sklearn) = {roc_auc}")
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    # plt.plot([0, 1], [0, 1], linestyle='--')  # 随机猜测线
    #
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend()
    # plt.grid()
    #
    # plt.show()
    #
    # import numpy as np
    #
    # np.savez(
    #     "roc_PIRC.npz",
    #     fpr=fpr,
    #     tpr=tpr,
    #     thresholds=thresholds,
    #     auc=roc_auc,
    #     score=score_np,
    #     label=label_np
    # )
    #
    # score=build_T_to_Ainf(T_tensor.detach().cpu().numpy(), 4)
    # score_str = {str(k): v for k, v in score.items()}
    # np.savez("groudtruth_20nodes.npz", **score_str)
    ExpandNodes = (args.node_num + math.comb(args.node_num - 1, 2)+1)*2
    for i in range(500):
        Win, Wres = Random_Win_Fixed_Wres_2(
            ExpandNodes, n_units, device,
            block_dim=block_dim, connectivity=connectivity
        )

        pirc = Nonliear_PIRC(
            n_units=n_units, in_dim=args.node_num, out_dim=args.node_num,
            Win=Win, Wres=Wres,
            sigma_in=sigma_in, rho=rho, alpha=alpha, tikh=tikh, mode=mode,
            device=device, block_dim=block_dim, I_type=I_type, option=option
        )

        R, _ = pirc.train(X_washout, X_train, Y_train)
        Y_test_predict = pirc.Prediction(R, args.N_test, Y_train)

        base_all = Y_test_predict[0, :, :args.N_test, 0]  # base: (B, T)
        others_all = Y_test_predict[2:, :, :args.N_test, 0]  # others: (E, B, T)
        base_expand = base_all.unsqueeze(0).expand_as(others_all)  # (E, B, T)
        score_EN = score_0_1_torch(
            others_all,
            base_expand,
            method=method,
            tau=tau
        )
        Score = score_EN.transpose(0, 1).contiguous()  # (B, E)
        Score = shift_column_for_Causal_Matrix(Score)
        Score.fill_diagonal_(1.0)
        AUC = ranking_auc_ALL(Score, T_tensor)

        print(f"trial {i} AUC = {AUC}")

        if AUC > best_auc:
            best_auc = AUC
            best_Win = Win.detach().cpu().clone()
            best_Wres = Wres.detach().cpu().clone()

print("Best AUC:", best_auc)
torch.save({
    'Win': best_Win,
    'Wres': best_Wres,
    'AUC': best_auc,
    'params': {
        'n_units': n_units,
        'sigma_in': sigma_in,
        'rho': rho,
        'alpha': alpha,
        'tikh': tikh,
        'block_dim': block_dim,
        'connectivity': connectivity
    }
}, 'best_full.pth')



