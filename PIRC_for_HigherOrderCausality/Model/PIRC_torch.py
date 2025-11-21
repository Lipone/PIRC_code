import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
import torch

class PIRC_Torch:
    def __init__(self, n_units, in_dim, out_dim, Structured_W, Win, Wres, sigma_in=0.5, rho=0.9, alpha=0.9, tikh=1e-4, option=1, device= None, block_dim = 1, use_Sin = False, use_Cos = False, mode = 0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_units = n_units
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma_in = sigma_in
        self.rho = rho
        self.tikh = tikh
        self.alpha = alpha
        self.option = option
        self.block_dim = block_dim
        self.ExpandDim = (math.comb(self.in_dim - 1, 2) * 2 + math.comb(self.in_dim, 1) * 2) * self.block_dim
        self.ExpandNodes = self.in_dim + math.comb(self.in_dim - 1, 2)
        self.Structured_W, self.Win, self.Wres = Structured_W, Win, Wres
        eigs = torch.linalg.eigvals(self.Wres)
        self.Wres = self.Wres * (rho / eigs.abs().max())
        self.n_units = self.Wres.shape[1]
        self.Wout = torch.zeros(self.n_units, self.out_dim, device=self.device)
        self.use_Sin = use_Sin
        self.use_Cos = use_Cos
        self.mode = mode

    def dataProcess(self, X, option=1):
        rows, cols = X.shape
        X_aug = torch.cat([torch.ones((rows, 1), device=self.device), X], dim=1)
        if  self.block_dim == 2:
            X = torch.cat([torch.ones((rows, 2), device=self.device), sin_cos_theta(X)], dim=1) @ self.Structured_W
            # X = sin_cos_theta(X) @ self.Structured_W
        elif self.block_dim == 1 and self.use_Sin==True:
            X = sin_theta(X_aug) @ self.Structured_W
        else:
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
        # if mode == 0:
        if A is not None:
            x = self.dataProcess(x, self.option) * A
        else:
            x = self.dataProcess(x, self.option)
        r_post = self.alpha * torch.tanh((x * self.sigma_in) @ self.Win.T + r_pre @ self.Wres.T) + (1 - self.alpha) * r_pre
        # if mode == 1:
        #     x = self.dataProcess(x, self.option)
        #     r_post = self.alpha * torch.tanh((x * self.sigma_in) @ self.Win.T + r_pre @ self.Wres.T) + (1 - self.alpha) * r_pre
        #     if A is not None:
        #         A = torch.as_tensor(A, device=self.device, dtype=r_post.dtype)
        #         rep = max(1, r_post.numel() // (A.numel()/self.ExpandNodes)) if A.numel() > 0 else 1
        #         if rep > 1:
        #             A = A.repeat(rep)
        #         A = A[:r_post.numel()].to(device=self.device, dtype=r_post.dtype)
        #         r_post = r_post * A
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
        RHS = R[1:].T @ Y_train[:,:self.out_dim]
        self.Wout = torch.linalg.solve(LHS, RHS)
        Y_pred = R[1:] @ self.Wout
        train_Loss = mean_squared_error(Y_train[:,:self.out_dim].cpu().numpy(), Y_pred.cpu().numpy())
        return R[1:], train_Loss, Y_pred.cpu().numpy()

    def evolve(self, r0, N_evo, Y_train = None, Y_test = None):
        R = torch.empty((N_evo + 1, self.n_units), device=self.device)
        R[0] = r0
        Yh = torch.zeros((N_evo + 1, self.in_dim), device=self.device)
        if self.out_dim == 1:
            Yh[0] = torch.cat([R[0] @ self.Wout, Y_train[-1, 1:]], dim=-1)
            for i in range(1, N_evo + 1):
                R[i] = self.step(R[i - 1], Yh[i - 1])
                Yh[i] = torch.cat([R[i] @ self.Wout, Y_test[i - 1, 1:]], dim=-1)
        else:
            Yh[0] = R[0] @ self.Wout
            for i in range(1, N_evo + 1):
                R[i] = self.step(R[i - 1], Yh[i - 1])
                Yh[i] = R[i] @ self.Wout
        return Yh[1:].cpu().numpy()

    def evolve_edge_removal(self, r0, j, i, N_evo, Y_train = None, Y_test = None):
        R = torch.zeros((N_evo + 1, self.n_units), device=self.device)
        R_intervened = torch.empty((N_evo + 1, self.n_units), device=self.device)
        R[0] = r0
        R_intervened[0] = r0
        A_intervened_in = torch.ones(self.ExpandDim, dtype=torch.float32, device=self.device)
        A_intervened_in[2 * j * self.block_dim:2 * j * self.block_dim + 2 * self.block_dim] = 0
        A_intervened_R = torch.ones_like(r0, dtype=torch.float32, device=self.device)
        A_intervened_R[ j * (A_intervened_R.numel() // self.ExpandNodes): (j + 1) * (A_intervened_R.numel() // self.ExpandNodes)] = 0
        Aout = torch.ones(self.out_dim, dtype=torch.float32, device=self.device)
        Aout[i] = 0
        A_intervened_out = torch.zeros(self.out_dim, dtype=torch.float32, device=self.device)
        A_intervened_out[i] = 1
        Yh = torch.zeros((N_evo + 1, self.in_dim), device=self.device)
        if self.out_dim == 1:
            Yh[0] = torch.cat([R_intervened[0] @ self.Wout, Y_train[-1, 1:]], dim=-1)
            for k in range(1, N_evo + 1):
                R_intervened[k] = self.step(R_intervened[k - 1], Yh[k - 1], A_intervened_in)
                Yh[k] = torch.cat([R_intervened[k] @ self.Wout, Y_test[k - 1, 1:]], dim=-1)
        else:
            Yh[0] = R[0] @ self.Wout
            for k in range(1, N_evo + 1):
                R[k] = self.step(R[k - 1], Yh[k - 1])
                if self.mode == 0:
                    R_intervened[k] = self.step(R_intervened[k - 1], Yh[k - 1], A_intervened_in)
                    Yh[k] = R[k] @ self.Wout * Aout + R_intervened[k] @ self.Wout * A_intervened_out
                if self.mode == 1:
                    # test = R[k] * A_intervened_R
                    Yh[k] = R[k] @ self.Wout * Aout + (R[k] * A_intervened_R)  @ self.Wout * A_intervened_out
        return Yh[1:].cpu().numpy()

    def Prediction(self, Y_test, R, N_test=100, Y_train=None):
        Loss_list = []
        Prediction_list = []
        r0 = R[-1]
        for j in range(self.ExpandNodes + 1):
            if j == 0:
                pred = self.evolve(r0, N_test, Y_train, Y_test)
            else:
                pred = self.evolve_edge_removal(r0, j - 1, 0, N_test, Y_train, Y_test)
            Prediction_list.append(pred)
            # Loss_list.append(nmse(Y_test[:N_test, 0].cpu().numpy(), pred[:, 0]))
            Loss_list.append(mean_squared_error(pred[:, 0], Y_test[:N_test, 0].cpu().numpy()))
        return Prediction_list, Loss_list
