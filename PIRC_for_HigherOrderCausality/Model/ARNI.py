#!/usr/bin/env python
import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from math import pi
import numpy as np
from sklearn.metrics import roc_curve, auc
from itertools import combinations

def ARNI(X, Y, BASIS, ORDER, NODE, connectivity,th=0.0001):
    # X：input time series data, shape (N, T)
    # Y: derivative of X, shape (N, T)
    # BASIS: type of basis expansion
    # ORDER: number of basis functions
    # NODE: target node for reconstruction
    # connectivity: ground truth connectivity matrix for evaluation
    # th: threshold for stopping criterion
    N, T = X.shape
    X = basis_expansion(X, ORDER, BASIS, NODE)
    nolist = list(range(N))
    llist = []
    cost = []
    b = 1
    vec = np.zeros(N, )
    while (nolist and (b == 1)):
        # composition of inferred subspaces
        Z = np.array([])
        for n in range(len(llist)):
            Z = np.vstack((Z, X[:, :, llist[n]])) if Z.size else X[:, :, llist[n]]

        # projection on remaining composite spaces
        P = np.zeros((len(nolist), 2))
        cost_err = np.zeros(len(nolist), )
        for n in range(len(nolist)):
            # composition of a possible spaces
            R = np.vstack((Z, X[:, :, nolist[n]])) if Z.size else X[:, :, nolist[n]]
            # error of projection on possible composite space
            # ( A.R=Y)
            RI = np.linalg.pinv(R)
            A = np.dot(Y[NODE, :], RI)
            Y_est = np.dot(A, R)
            DIFF = Y[NODE, :] - Y_est
            P[n, 0] = np.std(DIFF)  # the uniformity of error
            P[n, 1] = int(nolist[n])
            # Fitting cost of possible composite space
            cost_err[n] = (1 / T) * np.linalg.norm(DIFF)
            R = np.array([])

        # break if all candidates equivalent
        if np.std(P[:, 0]) < th:
            b = 0
            break

        else:
            # Selection of composite space which minimises projection error
            MIN = np.min(P[:, 0])  # best score
            block = np.argmin(P[:, 0])  # node index of best
            llist.append(int(P[block, 1]))  # add best node ID to llist
            nolist.remove(int(P[block, 1]))  # remove best from candidate list
            vec[int(P[block, 1])] = MIN  # used in ROC curve
            cost.append(cost_err[block])  # record SS Error
    # print('Reconstruction has finished!')

    if not llist:
        # print('WARNING: no predicted regulators - check that NODE abundance varies in the data!')
        pass
    elif connectivity is not None:
        # load connectivity for comparison
        adjacency = connectivity.copy()
        adjacency[adjacency != 0] = 1

        # print('Quality of reconstruction:')

        if (np.sum(adjacency[NODE, :]) == 0):
            # print('WARNING: no true regulators!')
            pass
        else:
            # Evaluation of results via AUC score
            from sklearn.metrics import roc_curve, auc

            FPR, TPR, _ = roc_curve(
                np.abs(adjacency[NODE, :]),
                np.abs(vec),
                pos_label=1
            )

    return llist, cost, np.abs(vec)

def basis_expansion(X, K, TYPE, NODE):
    N,M = X.shape
    Expansion = np.zeros((K+1, M, N))

    if TYPE == 'polynomial':
        for n in range(N):
            for k in range(K):
                Expansion[k,:,n] = X[n,:]**k

    elif TYPE == 'polynomial_diff':
        Xi =np.zeros((N,M))
        for m in range(M):
            Xi[:,m] = X[:,m]-X[NODE,m]

        for n in range(N):
            for k in range(K):
                Expansion[k,:,n] = Xi[n,:]**k

    elif TYPE == 'fourier':
        Expansion=np.zeros((2*(K), M,N))
        for n in range(N):
            t = 0
            for k in range(K):
                Expansion[k+t,:,n] = np.sin(k*X[n,:])
                Expansion[k+t+1,:,n] = np.cos(k*X[n,:])
                t += 1

    elif TYPE == 'fourier_diff':
        Expansion=np.zeros((2*(K),M,N))
        Xi = np.zeros((N,M))
        for m in range(M):
            Xi[:,m] = X[:,m] - X[NODE, m]

        for n in range(N):
            t = 0
            for k in range(K):
                Expansion[k+t,:,n] = np.sin(k*Xi[n,:])
                Expansion[k+t+1,:,n] = np.cos(k*Xi[n,:])
                t += 1

    elif TYPE == 'power_series':
        Expansion = np.zeros(((K)*(K), M, N))
        for n in range(N):
            for k1 in range(K):
                for k2 in range(K):
                    for m in range(M):
                        Expansion[((K)*k1)+k2, m, n] = (X[NODE,m]**k1)*(X[n,m]**k2)

    elif TYPE == 'RBF':
        Expansion = np.zeros((K,M,N))
        for n in range(N):
            A = np.vstack((X[n,:], X[NODE,:]))
            for m1 in range(K):
                for m2 in range(M):
                    Expansion[m1,m2, n] = np.sqrt(2.0+np.linalg.norm(A[:,m1]-A[:,m2],2)**2)

    return(Expansion)


def expand_nodes(X, NODE, type='add', order=3):
    n = X.shape[1]
    com_list = list(combinations([x for x in range(n) if x != NODE], 2))
    new_cols = []
    for j, k in com_list:
        if type == 'add':
            new_cols.append((X[:, j] + X[:, k]).reshape(-1,1))
        elif type == 'mul':
            new_cols.append((X[:, j] * X[:, k]).reshape(-1,1))
    if order == 4:
        com_list = list(combinations([x for x in range(n) if x != NODE], 3))
        for a,b,c in com_list:
            if type == 'add':
                new_cols.append((X[:, a] + X[:, b] + X[:, c]).reshape(-1, 1))
            elif type == 'mul':
                new_cols.append((X[:, a] * X[:, b] * X[:, c]).reshape(-1, 1))
    if new_cols:
        X = np.hstack([X] + new_cols)
    return X

def run_node(NODE, typ, base, order, expanded_cache, Y_T, T_Matrix=None):
    X_expanded = expanded_cache[(NODE, typ)]

    llist, cost, vec = ARNI(
        X_expanded.T,
        Y_T,
        base,
        order,
        NODE,
        T_Matrix,
        th=0.0001
    )

    return NODE, vec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_num', type=int, default=10, help='Number of nodes in the system')
    parser.add_argument('--Pair_strength', type=float, default=0.4, help='Pairwise interaction strength') #1
    parser.add_argument('--Tri_strength', type=float, default=0.4, help='Three-way interaction strength') #1
    parser.add_argument("--N_train", type=int, default=10000)
    parser.add_argument("--N_test", type=int, default=100)
    parser.add_argument("--N_washout", type=int, default=100)
    parser.add_argument("--N_start", type=int, default=1000)
    parser.add_argument("--probability", type=float, default=0.1) #0.02
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()
    set_seed(42)

    ExpandNodes = args.node_num + math.comb(args.node_num - 1, 2)

    data, a2, a3 = load_or_generate_kuramoto(args)
    data = data[args.N_start:,]

    T_Matrix = np.zeros((args.node_num, ExpandNodes))
    T_Matrix[:, :args.node_num] = (a2 != 0).astype(int)
    for i in range(args.node_num):
        com_list = list(combinations([x for x in range(args.node_num) if x != i], 2))
        for index, (j, k) in enumerate(com_list):
            T_Matrix[i, args.node_num + index] = int(a3[i][j][k] != 0)
    print("ground_truth", T_Matrix)

    X1 = data[:-1, :]
    X2 = data[1:, :]
    X = (X1+X2)*0.5
    # X = data[:-1, :] % (2 * np.pi)
    Y = (data[1:, :]- data[:-1, :])/args.dt

    n = X.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    for i in range(n):
        axes[i].plot(X[:, i])
        axes[i].set_ylabel(f'Node {i} Phase')
        axes[i].set_title(f'Node {i}')
    axes[-1].set_xlabel('Time step')
    plt.tight_layout()
    # plt.show()

    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    for i in range(n):
        axes[i].plot(Y[:, i])
        axes[i].set_ylabel(f'Node {i} Velocity')
        axes[i].set_title(f'Node {i}')
        axes[i].set_ylim(-5, 5)
    axes[-1].set_xlabel('Time step')
    plt.tight_layout()
    # plt.show()

    # 3) Run THIS
    lam_list1 = np.logspace(-1, 1, 50)  # 大 lambda 粗扫
    lam_list2 = np.logspace(-8, -1, 200)  # 小 lambda 密集扫
    lam_list = np.concatenate([lam_list1, lam_list2])[::-1]  # 倒序，先大 lambda
    best_lam, best_rho, best_auc, auc_grid = tune_lam_rho(
        X, Y, ooi, dmax, T_Matrix, args, ExpandNodes,
        lam_list=lam_list,  # λ 从 1e-6 到 10，共 15 个候选
        rho_list=np.logspace(-6, 0, 15),  # ρ 从 1e-6 到 1，共 15 个候选
        niter=50
    )

    Ainf, coeff, relerr = this(X.T, Y.T, ooi, dmax, lam=best_lam, rho=best_rho, niter=50)

    T_from_Ainf = np.zeros((args.node_num, ExpandNodes))
    com_dict = {}
    for i in range(args.node_num):
        com_list = list(combinations([x for x in range(args.node_num) if x != i], 2))
        com_dict[i] = {tuple(c): idx for idx, c in enumerate(com_list)}
    for order, mat in Ainf.items():
        if order == 2:
            for item in Ainf[order]:
                i0, j0, index = item[0], item[1], item[2]
                T_from_Ainf[int(i0), int(j0)] = abs(index)
        elif order == 3:
            for item in Ainf[order]:
                i0, j0, k0, index = item[0], item[1], item[2], item[3]
                # com_list = list(combinations([x for x in range(args.node_num) if x != i0], 2))
                idx = com_dict[i0][tuple(sorted((j0, k0)))]
                T_from_Ainf[int(i0), int(args.node_num + idx)] = abs(index)
    # AUC = ranking_auc_ALL(T_from_Ainf, T_Matrix)
    # print("AUC:", AUC)

    N = args.node_num
    mask = np.ones_like(T_Matrix, dtype=bool)
    mask[:N, :N] &= ~np.eye(N, dtype=bool)
    y_true = T_Matrix[mask]
    y_score = T_from_Ainf[mask]
    # y_true = T_Matrix.flatten()
    # y_score = T_from_Ainf.flatten()

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")  # 随机分类器
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

    print("AUC:", roc_auc)




