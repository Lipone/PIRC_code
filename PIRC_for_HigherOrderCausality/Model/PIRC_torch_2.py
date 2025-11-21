import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
import torch

class PIRC_Torch:
    def __init__(self, n_units, in_dim, out_dim, Structured_W, Win, Wres, sigma_in=0.5, rho=0.9, alpha=0.9, tikh=1e-4, option=1, device= None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_units = n_units
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma_in = sigma_in
        self.rho = rho
        self.tikh = tikh
        self.alpha = alpha
        self.option = option
        self.ExpandDim = math.comb(self.in_dim - 1, 2) * 2 + math.comb(self.in_dim, 1) * 2
        self.ExpandNodes = self.in_dim + math.comb(self.in_dim - 1, 2)
        self.Structured_W, self.Win, self.Wres = Structured_W, Win, Wres
        eigs = torch.linalg.eigvals(self.Wres)
        self.Wres = self.Wres * (rho / eigs.abs().max())
        self.Wout = torch.zeros(self.n_units, self.out_dim, device=self.device)

    def dataProcess(self, X, option=1):
        rows, cols = X.shape
        X_aug = torch.cat([torch.ones((rows, 1), device=self.device), X], dim=1)
        X = X_aug @ self.Structured_W
        if option == 0:
            X = X
        elif option == 1:
            X = X ** 2
        elif option == 2:
            X = torch.sin(X)
        elif option == 3:
            X = torch.tanh(X)
        elif option == 4:
            X = torch.sigmoid(X)
        elif option == 5:
            X = torch.relu(X)
        return X

    def step(self, r_pre, x, A=None):
        x = x.reshape(1, -1).to(self.device)
        if A is not None:
            x = self.dataProcess(x, self.option) * A
        else:
            x = self.dataProcess(x, self.option)
        r_post = self.alpha * torch.tanh((x * self.sigma_in) @ self.Win.T + r_pre @ self.Wres.T) + (1 - self.alpha) * r_pre
        return r_post

    def open_loop(self, X, r0):
        N = X.shape[0]
        R = torch.empty((N + 1, self.n_units), device=self.device)
        R[0] = r0
        for i in range(1, N + 1):
            R[i] = self.step(R[i - 1], X[i - 1])
        return R

    def train(self, X_washout, X_train, Y_train):
        rf_washout = self.open_loop(X_washout, torch.zeros(self.n_units, device=self.device))[-1]
        R = self.open_loop(X_train, rf_washout)
        LHS = R[1:].T @ R[1:] + self.tikh * torch.eye(self.n_units, device=self.device)
        RHS = R[1:].T @ Y_train
        self.Wout = torch.linalg.solve(LHS, RHS)
        Y_pred = R[1:] @ self.Wout
        train_Loss = mean_squared_error(Y_train.cpu().numpy(), Y_pred.cpu().numpy())
        return R[1:], train_Loss, Y_pred.cpu().numpy()

    def evolve(self, r0, N_evo, Y_train, Y_test):
        R = torch.empty((N_evo + 1, self.n_units), device=self.device)
        R[0] = r0
        Yh = torch.zeros((N_evo + 1, self.in_dim), device=self.device)
        Yh[0] = torch.cat([R[0] @ self.Wout, Y_train[-1, 1:]], dim=-1)
        for i in range(1, N_evo + 1):
            R[i] = self.step(R[i - 1], Yh[i - 1])
            Yh[i] = torch.cat([R[i] @ self.Wout, Y_test[i-1, 1:]], dim=-1)
        return Yh[1:].cpu().numpy()

    def evolve_edge_removal(self, r0, j, i, N_evo, Y_train, Y_test):
        R_intervened = torch.empty((N_evo + 1, self.n_units), device=self.device)
        R_intervened[0] = r0
        Yh = torch.zeros((N_evo + 1, self.in_dim), device=self.device)
        Yh[0] = torch.cat([R_intervened[0] @ self.Wout, Y_train[-1, 1:]], dim=-1)
        A_intervened_in = torch.ones(self.ExpandDim, dtype=torch.float32, device=self.device)
        A_intervened_in[2 * j:2 * j + 2] = 0
        for k in range(1, N_evo + 1):
            R_intervened[k] = self.step(R_intervened[k - 1], Yh[k - 1], A_intervened_in)
            Yh[k] = torch.cat([R_intervened[k] @ self.Wout, Y_test[k-1, 1:]], dim=-1)
        return Yh[1:].cpu().numpy()

    def Prediction(self, Y_test, R, N_test=100, Y_train=None):
        Loss_list = []
        Prediction_list = []
        r0 = R[-1]
        # print("R 状态矩阵：", R.size())
        # print("r0 初始状态：", r0)
        for j in range(self.ExpandNodes + 1):
            if j == 0:
                pred = self.evolve(r0, N_test, Y_train, Y_test)
            else:
                pred = self.evolve_edge_removal(r0, j - 1, 0, N_test, Y_train, Y_test)
            Prediction_list.append(pred)
            Loss_list.append(mean_squared_error(pred[:, 0], Y_test[:N_test, 0].cpu().numpy()))
        return Prediction_list, Loss_list
