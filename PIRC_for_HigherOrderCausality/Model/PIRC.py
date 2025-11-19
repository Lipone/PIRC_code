import numpy as np
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Library import *

class PIRC:
    def __init__(self, n_units, in_dim, out_dim, sigma_in=0.5, rho = 0.9, alpha=0.9, tikh=1e-4, option=1):
        self.n_units = n_units  #
        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.bias_in = 1.0  # input bias
        # self.bias_out = 1.0  # output bias
        self.sigma_in = sigma_in  # input scaling
        self.rho = rho  # spectral radius
        self.tikh = tikh  # Tikhonov factor
        self.alpha= alpha # leak rate
        self.ExpandDim=math.comb(self.in_dim-1, 2)*2 + math.comb(self.in_dim, 1)*2
        self.ExpandNodes= self.in_dim + math.comb(self.in_dim-1, 2)
        self.Win, self.Wres= self.Partitioned_Win_Wres(self.n_units)
        self.Wres = self.Wres* (rho / np.max(np.abs(np.linalg.eigvals(self.Wres))))
        self.option=option  # 1: structured input, 2: nonlinear input
        self.Structured_W = self.generate_structured_w(in_dim)


    def Partitioned_Win_Wres(self, n):
        # k : number of inputs
        # n : number of reservoir nodes
        # k+math.comb(k-1, 2) : number of blocks
        block_num = self.ExpandNodes
        Win_block_row = math.floor(n / block_num)
        Win_block_col = 2
        # 每个分块矩阵块的大小 row*col，每个分块矩阵元素满足-1至1均匀分布,生成block_num个分块矩阵，并拼接成一个对角矩阵
        Win_blocks = [rng.uniform(-1, 1, (Win_block_row, Win_block_col)) for _ in range(block_num)]
        Win = np.block(
            [[Win_blocks[i] if i == j else np.zeros((Win_block_row, Win_block_col)) for j in range(block_num)] for i in
             range(block_num)])
        Wres_block_row = Wres_block_col = Win_block_row
        Wres_blocks = [rng.uniform(-1, 1, (Wres_block_row, Wres_block_col)) for _ in range(block_num)]
        Wres = np.block(
            [[Wres_blocks[i] if i == j else np.zeros((Wres_block_row, Wres_block_col)) for j in range(block_num)] for i
             in
             range(block_num)])
        return Win, Wres

    def Partitioned_Win_Wres_sameBlocks(self, n):
        block_num = self.ExpandNodes
        Win_block_row = math.floor(n / block_num)
        Win_block_col = 2
        # 每个分块矩阵块的大小 row*col，每个分块矩阵元素满足-1至1均匀分布,生成block_num个分块矩阵，并拼接成一个对角矩阵
        Win_block = rng.uniform(-1, 1, (Win_block_row, Win_block_col))
        Win_blocks = [Win_block for _ in range(block_num)]
        Win = np.block(
            [[Win_blocks[i] if i == j else np.zeros((Win_block_row, Win_block_col)) for j in range(block_num)] for i in
             range(block_num)])
        Wres_block_row = Wres_block_col = Win_block_row
        Wres_block = rng.uniform(-1, 1, (Wres_block_row, Wres_block_col))
        Wres_blocks = [Wres_block for _ in range(block_num)]
        Wres = np.block(
            [[Wres_blocks[i] if i == j else np.zeros((Wres_block_row, Wres_block_col)) for j in range(block_num)] for i
             in
             range(block_num)])
        return Win, Wres

    def dataProcess(self, X, option=1):
        # 数据预处理，添加偏置项
        rows, cols = X.shape
        X_aug = np.hstack([np.ones((rows, 1)),X])
        if option ==0:
            X= np.dot(X_aug, self.Structured_W)
        if option ==1:
            X= (np.dot(X_aug, self.Structured_W))**2
        if option ==2:
            X = np.sin(np.dot(X_aug, self.Structured_W))
        if option ==3:
            X = np.tanh(np.dot(X_aug, self.Structured_W))
        return X

    def generate_structured_w(self, cols):
        Structured_W = np.zeros((cols + 1, self.ExpandDim), dtype=float)

        # Cause
        Structured_W[0, 0] = rng.uniform(-1, 1)
        Structured_W[1, 1] = rng.uniform(-1, 1)

        # Pairwise
        for p in range(1, cols):
            Structured_W[p + 1, p * 2] = rng.uniform(-1, 1)
            Structured_W[1, p * 2 + 1] = rng.uniform(-1, 1)
            Structured_W[p + 1, p * 2 + 1] = rng.uniform(-1, 1)

        # Tri-order
        index_list = list(itertools.combinations(range(cols - 1), 2))
        shifted_index_list = [(i + 2, j + 2) for i, j in index_list]
        for index, (q, w) in enumerate(shifted_index_list):
            Structured_W[1, (index + cols) * 2 + 1] = rng.uniform(-1, 1)
            Structured_W[q, (index + cols) * 2] = rng.uniform(-1, 1)
            Structured_W[q, (index + cols) * 2 + 1] = rng.uniform(-1, 1)
            Structured_W[w, (index + cols) * 2] = rng.uniform(-1, 1)
            Structured_W[w, (index + cols) * 2 + 1] = rng.uniform(-1, 1)

        return Structured_W

    def step(self, r_pre, x, A=None):
        if A is not None:
            x = self.dataProcess(x.reshape(1, -1)) * A
        else:
            x = self.dataProcess(x.reshape(1, -1))
        r_post = self.alpha*np.tanh(np.dot(x * self.sigma_in, self.Win.T) + np.dot(r_pre, self.Wres.T))+(1-self.alpha)*r_pre
        return r_post

    def open_loop(self, X, r0):
        N = X.shape[0]
        R = np.empty((N + 1, self.n_units))
        R[0, :] = r0
        for i in 1 + np.arange(N):
            R[i] = self.step(R[i - 1, :self.n_units], X[i - 1])
        return R

    def train(self, X_washout, X_train, Y_train):
        rf_washout = self.open_loop(X_washout, np.zeros(self.n_units))[-1, :self.n_units]
        ## open-loop train phase
        R = self.open_loop(X_train, rf_washout)
        ## Ridge Regression
        LHS = np.dot(R[1:].T, R[1:]) + self.tikh * np.eye(self.n_units)
        RHS = np.dot(R[1:].T, Y_train)
        self.Wout = np.linalg.solve(LHS, RHS)
        Y_pred = np.dot(R[1:], self.Wout)
        if np.isnan(Y_pred).any():
            return R[1:], 1e6, Y_pred
        train_Loss = mean_squared_error(Y_train, Y_pred)  # 计算均方误差
        return R[1:], train_Loss, Y_pred

    def train_single(self, X_washout, X_train, Y_train):
        rf_washout = self.open_loop(X_washout, np.zeros(self.n_units))[-1, :self.n_units]
        ## open-loop train phase
        R = self.open_loop(X_train, rf_washout)
        ## Ridge Regression
        LHS = np.dot(R[1:].T, R[1:]) + self.tikh * np.eye(self.n_units)
        RHS = np.dot(R[1:].T, Y_train)
        self.Wout = np.linalg.solve(LHS, RHS)
        Y_pred = np.dot(R[1:], self.Wout)
        if np.isnan(Y_pred).any():
            return R[1:], 1e6
        train_single_Loss = mean_squared_error(Y_train[:,1], Y_pred[:,1])  # 计算均方误差
        return R[1:], train_single_Loss, Y_pred

    def evolve(self, r0, N_evo):
        R = np.empty((N_evo + 1, self.n_units))
        R[0] = r0
        Yh = np.zeros((N_evo + 1, self.out_dim))
        Yh[0] = np.dot(R[0], self.Wout)
        for i in 1 + np.arange(N_evo):
            R[i] = self.step(R[i - 1, :self.n_units], Yh[i - 1])
            Yh[i] = np.dot(R[i], self.Wout)
        return Yh[1:]

    def evolve_edge_removal(self, r0, j, i, N_evo):
        # 干预j，看i的变化
        R = np.empty((N_evo + 1, self.n_units))
        R_intervened = np.empty((N_evo + 1, self.n_units))
        R[0] = r0
        R_intervened[0] = r0
        Yh = np.zeros((N_evo + 1, self.out_dim))
        Yh[0] = np.dot(R[0], self.Wout)
        A_intervened_in = np.ones(self.ExpandDim)  # 0_j
        A_intervened_in[2*j:2*j+2] = 0
        Aout = np.ones(self.out_dim)
        Aout[i] = 0
        A_intervened_out = np.zeros(self.out_dim)  # 1_i
        A_intervened_out[i] = 1
        for i in 1 + np.arange(N_evo):
            R[i] = self.step(R[i - 1, :self.n_units], Yh[i - 1])
            R_intervened[i] = self.step(R_intervened[i - 1, :self.n_units], Yh[i - 1], A_intervened_in)
            Yh[i] = np.dot(R[i], self.Wout)* Aout + np.dot(R_intervened[i], self.Wout) * A_intervened_out
        return Yh[1:]

    def Prediction(self, Y_test, R, N_test=100):
        Loss_list = []
        Prediction_list= []
        r0=R[-1:,]
        for j in range(self.ExpandNodes+1):
            if j==0:
                Prediction_list.append(self.evolve(r0, N_test))
            else:
                Prediction_list.append(self.evolve_edge_removal(r0, j-1, 0, N_test))
            Loss_list.append(mean_squared_error(Prediction_list[j][:,0], Y_test[:,0]))
        return Prediction_list, Loss_list





