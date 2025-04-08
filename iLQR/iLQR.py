import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import os
from CartPoleEnv import CartPoleILQREnv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.autograd.set_detect_anomaly(True)

def gradient(f, x, delta=1e-5):
    (n,) = x.shape
    return np.array([(f(x + delta * np.eye(1, n, i).flatten())- f(x - delta * np.eye(1, n, i).flatten()))/ (2 * delta) for i in range(n)])


def jacobian(f, x, delta=1e-5):
    (n,) = x.shape
    (m,) = f(x).shape
    x = x.astype("float32")
    return np.array([gradient(lambda x: f(x)[i], x, delta) for i in range(m)])


def hessian(f, x, delta=1e-5):
    return jacobian(lambda x: gradient(f, x, delta), x, delta)


def c(env, s, a):
    assert s.shape == (4,)
    assert a.shape == (1,) 
    
    env.reset()
    env.unwrapped.state = s.copy()
    
    observation, reward, terminated, truncated, info = env.step(a)
    cost = -reward
    
    return cost


def f(env, s, a):
    assert s.shape == (4,)
    assert a.shape == (1,)
    
    env.reset()
    env.unwrapped.state = s.copy()
    
    next_observation, reward, terminated, truncated, info = env.step(a)
    
    return next_observation


def compute_local_expansions(env,s, a):
    return (
        jacobian(lambda x: f(env,x, a), s),
        jacobian(lambda x: f(env,s, x), a),
        hessian(lambda x: c(env,x, a), s),
        hessian(lambda x: c(env,s, x), a),
        hessian(lambda x: c(env,x[: len(s)], x[len(s) :]), np.concatenate((s, a)))[
            : len(s), [len(s)]
        ],
        gradient(lambda x: c(env,x, a), s),
        gradient(lambda x: c(env,s, x), a),
    )

def compute_Q_params(A, B, Q, R, M, q, r, m, b, P, y, p):
    C = Q + (A.T @ P @ A)
    D = R + (B.T @ P @ B)
    E = M + (2 * (A.T @ P @ B))
    ft = q.T + (2 * (m.T @ P @ A)) + (y.T @ A)
    gt = r.T + (2 * (m.T @ P @ B)) + (y.T @ B)
    h = (m.T @ P @ m) + (y.T @ m + p) + b

    return C, D, E, ft.T, gt.T, h.flatten()

def compute_policy(A, B, m, C, D, E, f, g, h):
    D_inv = np.linalg.inv(D)
    K = -0.5 * (D_inv @ E.T)
    k = -0.5 * (D_inv @ g)
    return K, k.flatten()

def compute_V_params(A, B, m, C, D, E, f, g, h, K, k):
    P = C + (K.T @ D @ K) + (E @ K)
    yt = (2 * (k.T @ D @ K)) + (k.T @ E.T) + (g.T @ K) + f.T
    p = (k.T @ D @ k) + (g.T @ k) + h

    return P, yt.T, p.flatten()

def ilqr(env, s_star, a_star, T):
    N_s = s_star.shape[0]
    N_a = a_star.shape[0]

    # get A, B, Q, R, M, q, r
    A, B, Q, R, M, q, r = compute_local_expansions(env, s_star, a_star)

    # Create H block and make PD
    H = np.concatenate((np.concatenate((Q, M.T)), np.concatenate((M, R))), axis=1)
    H_eval, H_evec = np.linalg.eig(H)
    H = sum(
        [
            (
                H_eval[i] * np.outer(H_evec[i], H_evec[i])
                if H_eval[i] > 0
                else np.zeros((N_s + N_a, N_s + N_a))
            )
            + 1e-5 * np.eye(N_s + N_a)
            for i in range(N_s + N_a)
        ]
    )

    # Extract updated Q_2, M, R_2, q_2, r_2, b, m
    Q, R, M = H[:N_s, :N_s], H[N_s:, N_s:], H[:N_s, N_s:]
    Q_2, R_2, q_2, r_2, b, m = (
        Q / 2,
        R / 2,
        (q.T - s_star.T @ Q - a_star.T @ M.T).T,
        (r.T - a_star.T @ R - s_star.T @ M).T,
        c(env,s_star, a_star)
        + 0.5 * (s_star.T @ (Q / 2) @ s_star + a_star.T @ R @ a_star)
        + s_star.T @ M @ a_star
        - q.T @ s_star
        - r.T @ a_star,
        (f(env,s_star, a_star) - A @ s_star - B @ a_star).reshape(
            -1, 1
        ),
    )

    # Compute K, k with base step time t = T-1
    policy = [
        (
            -0.5 * np.linalg.inv(R_2) @ M.T,
            (-0.5 * np.linalg.inv(R_2) @ r_2).flatten(),
        )
    ]

    # Compute parameters of V_{T-1}^{star}
    (
        P,
        y,
        p,
    ) = (
        Q_2 - 0.25 * M @ np.linalg.inv(R_2) @ M.T,
        (q_2.T - 0.5 * r_2.T @ np.linalg.inv(R_2) @ M.T).T,
        (b - 0.25 * r_2.T @ np.linalg.inv(R_2) @ r_2).flatten(),
    )

    # Loop through other time steps inductively
    for _ in range(T - 2, -1, -1):
        # Compute parameters of Q_t^{star}
        C, D, E, f1, g, h = compute_Q_params(
            A, B, Q_2, R_2, M, q_2, r_2, m, b, P, y, p
        )

        # Compute K_t, k_t and add it to policy list
        K_t, k_t = compute_policy(_, _, _, _, D, E, _, g, _)
        policy = [(K_t, k_t)] + policy

        # Compute parameters of V_t^{star}
        P, y, p = compute_V_params(_, _, _, C, D, E, f1, g, h, K_t, k_t)

    return policy

def test(episodes,policies,init_state,render=True):
    # Create FrozenLake instance
    env = CartPoleILQREnv( render_mode='human' if render else None)

    env.reset()
    env.unwrapped.state = init_state.copy()
    state = init_state.copy()
    for t in range(episodes):
        env.render()  
        (K, k) = policies[t]
        action = K @ state + k
        state, reward, done, _, _ = env.step(action)
        if done:
            print("end!")
            break

    env.close()

if __name__ == '__main__':
    env = CartPoleILQREnv()

    # env.close()
    s_star = np.array([0, 0, 0, 0], dtype=np.float32)
    a_star = np.array([0], dtype=np.float32)
    # print(f(env,s_star,a_star))
    A, B, Q, R, M, q, r = compute_local_expansions(env,s_star, a_star)
    T = 500
    policies = ilqr(env, s_star, a_star, T)
    init_state =  np.array([0.0, 0.0, 0.2, 0.0])
    test(T,policies,init_state)