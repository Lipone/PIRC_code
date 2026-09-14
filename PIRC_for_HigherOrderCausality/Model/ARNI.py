#!/usr/bin/env python
import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from math import pi
import numpy as np
from sklearn.metrics import roc_curve, auc
from itertools import combinations

def ARNI(X, Y, BASIS, ORDER, NODE, connectivity=None,th=0.0001):
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

