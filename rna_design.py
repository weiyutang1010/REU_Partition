import os
import sys
import time
import argparse
import json
import numpy as np
import math
from collections import defaultdict
from util import dicts_to_lists, generate_sequences, generate_rand_structure, generate_rand_distribution, probability, verify_dicts, RNA
import plotext as plt

RT = 1.
nucs = 'ACGU'
_allowed_pairs = [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)] 
_invalid_pairs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)]
NEG_INF = float('-1e18')
SMALL_NUM = float('1e-18')

def paired(c1, c2):
    _allowed_pairs = {"CG": -3, "GC": -3, "AU": -2, "UA":-2, "GU": -1, "UG":-1}
    c1, c2 = nucs[c1], nucs[c2]
    if c1 + c2 in _allowed_pairs:
        return _allowed_pairs[c1 + c2]
    else:
        return 10.

def unpaired(c='A'):
    return 1. # for now set every unpaired to same score

def projection_simplex_np_batch(x, z=1): # Tian Shuo's projection code
    x_sorted = np.sort(x, axis=1)[:, ::-1]
    cumsum = np.cumsum(x_sorted, axis=1)
    denom = np.arange(x.shape[1]) + 1
    theta = (cumsum - z)/denom
    mask = x_sorted > theta 
    csum_mask = np.cumsum(mask, axis=1)
    index = csum_mask[:, -1] - 1
    x_proj = np.maximum(x.transpose() - theta[np.arange(len(x)), index], 0)
    return x_proj.transpose()

def expected_inside_partition_log(X, sharpturn=0):
    n = len(X)
    Q_hat = defaultdict(lambda: defaultdict(lambda: NEG_INF))

    for j in range(n):
        Q_hat[j-1][j] = 0.

    for j in range(n):
        for i in Q_hat[j-1]:
            unpaired_sc = NEG_INF
            for nucj in range(4):
                unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j][nucj] + SMALL_NUM) + (-unpaired(nucj) / RT))
            Q_hat[j][i] = np.logaddexp(Q_hat[j][i], Q_hat[j-1][i] + unpaired_sc)

            if i > 0 and j-(i-1) > sharpturn:
                paired_sc = NEG_INF # calculate weighted paired score
                for nuci_1, nucj in _allowed_pairs:
                    paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][nuci_1] + SMALL_NUM) + np.log(X[j][nucj] + SMALL_NUM) + (-paired(nuci_1, nucj)))
                
                for k in Q_hat[i-2]:
                    Q_hat[j][k] = np.logaddexp(Q_hat[j][k], Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc)

    return Q_hat


def expected_outside_partition_log(Q_hat, X, sharpturn=0):
    """Backpropagation of Expected Inside Partition. Q_hat, O_hat, and gradient are calculated in log space.
       The dynet result is given by exp(gradient)."""
    n = len(X)
    O_hat = defaultdict(lambda: defaultdict(lambda: NEG_INF))
    gradient = np.array([[NEG_INF, NEG_INF, NEG_INF, NEG_INF] for _ in range(n)])

    O_hat[n-1][0] = 0.
    for j in range(n-1, -1, -1):
        for i in Q_hat[j-1]:
            unpaired_sc = NEG_INF # calculate weighted unpaired score
            for nucj in range(4):
                unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j][nucj] + SMALL_NUM) + (-unpaired(nucj) / RT))
            O_hat[j-1][i] = np.logaddexp(O_hat[j-1][i], O_hat[j][i] + Q_hat[j-1][i] + unpaired_sc - Q_hat[j][i])

            for nucj in range(4):
                # gradient of Q_hat[1][n] with respect to unpaired_sc
                grad = O_hat[j][i] + Q_hat[j-1][i] + unpaired_sc - Q_hat[j][i]
                # gradient of unpaired_sc with respect to nucj
                softmax = np.log(X[j][nucj] + SMALL_NUM) + (-unpaired(nucj) / RT) - unpaired_sc
                gradient[j][nucj] =  np.logaddexp(gradient[j][nucj], grad - np.log(X[j][nucj] + SMALL_NUM) + softmax)

            if i > 0 and j-(i-1) > sharpturn:
                paired_sc = NEG_INF # calculate weighted paired score
                for c1, c2 in _allowed_pairs:
                    paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][c1] + SMALL_NUM) + np.log(X[j][c2] + SMALL_NUM) + (-paired(c1, c2) / RT))

                for k in Q_hat[i-2]:
                    O_hat[i-2][k] = np.logaddexp(O_hat[i-2][k], O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k])
                    O_hat[j-1][i] = np.logaddexp(O_hat[j-1][i], O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k])

                    # gradient of Q_hat[1][n] with respect to paired_sc
                    grad = O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k]

                    # gradient of Q_hat[1][n] with respect to nuci_1, nucj
                    for nuci_1, nucj in _allowed_pairs:
                        softmax = np.log(X[i-1][nuci_1] + SMALL_NUM) + np.log(X[j][nucj] + SMALL_NUM) + (-paired(nuci_1, nucj)) - paired_sc
                        gradient[i-1][nuci_1] = np.logaddexp(gradient[i-1][nuci_1], grad - np.log(X[i-1][nuci_1] + SMALL_NUM) + softmax)
                        gradient[j][nucj] = np.logaddexp(gradient[j][nucj], grad - np.log(X[j][nucj] + SMALL_NUM) + softmax)

    return gradient


def expected_free_energy(struct, X):
    n = len(struct)
    free_energy = 0.
    penalty = 10.

    gradient = np.array([[0., 0., 0., 0.] for _ in range(n)])

    stack = []
    for j in range(n):
        if struct[j] == '(':
            stack.append(j)
        elif struct[j] == ')':
            i = stack.pop()

            score_ij = 0.
            
            for nuci, nucj in _allowed_pairs:
                score_ij += X[i][nuci] * X[j][nucj] * paired(nuci, nucj)
                gradient[i][nuci] += X[j][nucj] * paired(nuci, nucj)
                gradient[j][nucj] += X[i][nuci] * paired(nuci, nucj)

            for nuci, nucj in _invalid_pairs:
                score_ij += X[i][nuci] * X[j][nucj] * penalty
                gradient[i][nuci] += X[j][nucj] * penalty
                gradient[j][nucj] += X[i][nuci] * penalty

            free_energy += score_ij
        else:
            free_energy += unpaired()

    return free_energy, gradient

def gradient_descent(rna_struct, lr, num_step, sharpturn):
    n = len(rna_struct)

    log = []

    # starting distribution
    X = np.array([[.25, .25, .25, .25] for _ in range(n)])

    for epoch in range(num_step):

        log_Q_hat = expected_inside_partition_log(X, sharpturn)
        grad1 = expected_outside_partition_log(log_Q_hat, X, sharpturn)

        Delta_G, grad2 = expected_free_energy(rna_struct, X)

        objective_value = Delta_G + log_Q_hat[n-1][0]
        print(f'step: {epoch: 4d}, objective value: {objective_value:.8f}')
        log.append(objective_value)

        
        X = X - lr * (grad1 + grad2)
        X = projection_simplex_np_batch(X)

    for i in range(n):
        print(f"A: {X[i][0]:.4f}, C: {X[i][1]:.4f}, G: {X[i][2]:.4f}, U: {X[i][3]:.4f}")

    seq = "".join([nucs[index] for index in np.argmax(X, axis=1)])
    print("optimal sequence: ", seq)

    plt.plot(log)
    plt.plot_size(60, 20)
    plt.xlabel('step')
    plt.ylabel('Objective Value')
    plt.title(f'{rna_struct}: n={n}, lr = {lr}, step={len(log)}')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", '-s', type=str, default='')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--sharpturn", type=int, default=0)
    args = parser.parse_args()
    print("Args: ", end='')
    print(args)

    rna_struct = args.structure
    lr = args.lr
    num_step = args.step
    sharpturn = args.sharpturn

    gradient_descent(rna_struct, lr, num_step, sharpturn)

if __name__ == '__main__':
    main()

