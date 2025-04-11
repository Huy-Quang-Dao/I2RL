import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import os
from scipy.sparse import csr_matrix
from tqdm import tqdm
from CartPoleEnv import CartPoleILQREnv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.autograd.set_detect_anomaly(True)


# Differential Dynamic Programming
class DDP():
    def __init__(self,env,state_init,pred_time):
        super(DDP, self).__init__()
        self.env = env
        self.pred_time = pred_time
        self.state_init = state_init
        self.umax = 20
        self.N_s = env.observation_space.shape[0]
        self.f = lambda x, u: env._state_eq(x, u)  # dynamic
        self.l = lambda x, u: 0.5 * np.sum(np.square(u))  # l(x, u)
        self.lN = lambda x: 0.5 * (np.square(1.0 - np.cos(x[2])) + np.square(x[1]) + np.square(x[3]))  # final cost
        # self.v = [0.0 for _ in range(self.pred_time + 1)]
        self.p = [np.zeros(self.N_s) for _ in range(self.pred_time + 1)]
        self.P = [np.zeros((self.N_s, self.N_s)) for _ in range(self.pred_time + 1)]
        # expansion dynamics jacobian of jacobian
        self.f_x = lambda s, a: self.compute_local_expansions(
            s, a
        )[0]
        self.f_u = lambda s, a: self.compute_local_expansions(
            s, a
        )[1]
        self.f_xx = lambda s, a: self.jacobian_tensor(self.f_x, s, a, 0)  # |S|x|S|x|S|
        self.f_uu = lambda s, a: self.jacobian_tensor(self.f_u, s, a, 1)  # |S|x|A|x|A|
        self.f_ux = lambda s, a: self.jacobian_tensor(self.f_u, s, a, 0)  # |S|x|A|x|S|

    def gradient(self,fx, x, delta=1e-5):
        (n,) = x.shape
        return np.array([(fx(x + delta * np.eye(1, n, i).flatten())- fx(x - delta * np.eye(1, n, i).flatten()))/ (2 * delta) for i in range(n)])


    def jacobian(self,fx, x, delta=1e-5):
        (n,) = x.shape
        (m,) = fx(x).shape
        x = x.astype("float32")
        return np.array([self.gradient(lambda x: fx(x)[i], x, delta) for i in range(m)])


    def hessian(self,fx, x, delta=1e-5):
        return self.jacobian(lambda x: self.gradient(fx, x, delta), x, delta)
    
    def jacobian_tensor(self,f, x, a, index, delta=1e-5):
        m, n = f(x, a).shape
        x = x.astype("float64")
        if index == 0:
            (l,) = x.shape
            jacobian = np.zeros((m, n, l)).astype("float64")
            for i in range(n):
                x[i] += delta
                gplus = f(x, a)
                x[i] -= 2 * delta
                gminus = f(x, a)
                x[i] += delta
            jacobian[:, i] = (gplus - gminus) / (2 * delta)

        else:
            (l,) = a.shape
            jacobian = np.zeros((m, n, l)).astype("float64")
            for i in range(n):
                a[i] += delta
                gplus = f(x, a)
                a[i] -= 2 * delta
                gminus = f(x, a)
                a[i] += delta
            jacobian[:, i] = (gplus - gminus) / (2 * delta)

        return jacobian
    
    def compute_local_expansions(self,s, a):
        return (
            self.jacobian(lambda x: self.f(x, a), s),
            self.jacobian(lambda x: self.f(s, x), a),
            self.hessian(lambda x: self.l(x, a), s),
            self.hessian(lambda x: self.l(s, x), a),
            self.hessian(lambda x: self.l(x[: len(s)], x[len(s) :]), np.concatenate((s, a)))[
                : len(s), [len(s)]
            ],
            self.gradient(lambda x: self.l(x, a), s),
            self.gradient(lambda x: self.l(s, x), a),
        )
    
    def vec(self,A):
        m, n = A.shape[0], A.shape[1]
        return A.reshape(m*n, order='F')

    def commutation_matrix_sp(self,A):
        m, n = A.shape[0], A.shape[1]
        row  = np.arange(m*n)
        col  = row.reshape((m, n), order='F').ravel()
        data = np.ones(m*n, dtype=np.int8)
        K = csr_matrix((data, (row, col)), shape=(m*n, m*n))
        return K
    
    
    def backward(self, x_seq, u_seq):
        # Cost at N
        self.p[-1] = self.gradient(self.lN, x_seq[-1])  # p_N
        self.P[-1] = self.hessian(self.lN, x_seq[-1])  # P_N

        # Policy
        d_k = []
        K_k = []

        for t in range(self.pred_time - 1, -1, -1):
            compute_at_k = self.compute_local_expansions(x_seq[t], u_seq[t])
            A_k = compute_at_k[0]  # f_x_k
            B_k = compute_at_k[1]  # f_u_k
            l_xx = compute_at_k[2]
            l_uu = compute_at_k[3]
            l_xu = compute_at_k[4]
            l_x = compute_at_k[5]
            l_u = compute_at_k[6]

            Q_x = l_x + np.matmul(A_k.T, self.p[t + 1])  # \nabla_x(l) + A_t^T @ p_{k+1}
            Q_u = l_u + np.matmul(B_k.T, self.p[t + 1])  # \nabla_u(l) + B_t^T @ p_{k+1}

            Q_xx = (
                l_xx
                + np.matmul(np.matmul(A_k.T, self.P[t + 1]), A_k)
                + np.dot(self.p[t + 1].T, np.squeeze(self.f_xx(x_seq[t], u_seq[t])))
            )  # \nabla_xx(l) + A_k^T @ P_{t+1} @ A_k + (kron(p_{k+1},I)) @ T @ dAkdx
            Q_uu = (
                l_uu
                + np.matmul(np.matmul(B_k.T, self.P[t + 1]), B_k)
                + np.dot(self.p[t + 1].T, np.squeeze(self.f_uu(x_seq[t], u_seq[t])))
            ) # \nabla_uu(l) + B_k^T @ P_{t+1} @ B_k + (kron(p_{k+1},I)) @ T @ dBkdu

            Q_ux = (
                l_xu.T
                + np.matmul(np.matmul(B_k.T, self.P[t + 1]), A_k)
                + (np.dot(self.p[t + 1].T, np.squeeze(self.f_ux(x_seq[t], u_seq[t]))))
            )  # \nabla_xu(l) + A_k^T @ P_{t+1} @ B_k + (kron(p_{k+1},I)) @ T @ dAkdu

            inv_Q_uu = np.linalg.inv(Q_uu)
            d = np.matmul(inv_Q_uu, Q_u)
            K = np.matmul(inv_Q_uu, Q_ux)
            self.p[t] = Q_x - np.matmul(K.T,Q_u) + np.matmul(np.matmul(K.T,Q_uu),d) - np.matmul(Q_ux.T,d)
            self.P[t] = Q_xx + np.matmul(np.matmul(K.T,Q_uu),K) - np.matmul(Q_ux.T,K) - np.matmul(K.T,Q_ux)
            d_k.append(d)
            K_k.append(K)
        d_k.reverse()
        K_k.reverse()
        return d_k, K_k
    
    def forward(self, x_seq, u_seq, d_k, K_k):
        T, n_a, n_s = len(u_seq), len(u_seq[0]), len(x_seq[0])
        x_seq_hat, u_seq_hat = np.zeros((T + 1, n_s)), np.zeros((T, n_a))
        x_seq_hat[0] = x_seq[0]

        # Note: please clip the action within (-self.umax, self.umax), you can use np.clip
        for t in range(T):
            tmp = u_seq[t] - d_k[t] - K_k[t] @ (x_seq_hat[t] - x_seq[t])
            u_seq_hat[t] = np.clip(tmp, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])

        # return x_seq_hat and u_seq_hat
        return x_seq_hat, u_seq_hat
    

    def train(self,episodes):
        u_seq = [np.zeros(1) for _ in range(self.pred_time)]
        x_seq = [self.state_init.copy()]
        for t in range(self.pred_time):
            x_seq.append(self.env._state_eq(x_seq[-1], u_seq[t]))

        policys = []
        # for i in tqdm(range(episodes)):
        for i in range(episodes):
            # backward to calculate k_seq and kk_seq
            d_k,K_k = self.backward(x_seq, u_seq)

            # forward to calculate x_seq_hat and u_seq_hat
            x_seq, u_seq = self.forward(np.array(x_seq), np.array(u_seq), d_k, K_k)

            x_seq[0] = self.f(x_seq[0], u_seq[0])

            # append the policy for s_0 to policys
            policys.append(u_seq[0])

        return policys
    
    def test(self,episodes,policies,init_state,render=True):
    # Create FrozenLake instance
        env = CartPoleILQREnv( render_mode='human' if render else None)

        env.reset()
        env.unwrapped.state = init_state.copy()
        state = init_state.copy()
        for t in range(episodes):
            env.render()  
            action = policies[t]
            state, reward, done, _, _ = env.step(action)
            # if done:
            #     print("end!")
            #     break

        env.close()




if __name__ == '__main__':
    env = CartPoleILQREnv()
    s_star = np.array([0.0, 0.0, 0.5 * np.pi, 0.0], dtype=np.float32)
    a_star = np.array([0], dtype=np.float32)
    ddp = DDP(env,s_star,50)
    T = 400
    # policies = ddp.train(T)
    # np.save('DDP\policies2.npy', policies)
    policies = np.load('DDP\policies2.npy')
    # print(policies)
    ddp.test(T,policies,s_star)





















