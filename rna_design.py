import os
import sys
import time
import argparse
import json
import numpy as np
import math
from enum import Enum
from collections import defaultdict
from util import dicts_to_lists, generate_sequences, generate_rand_structure, generate_rand_distribution, probability, verify_dicts, RNA
import concurrent.futures

RT = 1.
nucs = 'ACGU'
nucpairs = ['CG', 'GC', 'AU', 'UA', 'GU', 'UG']
_allowed_pairs = [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)] 
_invalid_pairs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)]
NEG_INF = float('-1e18')
SMALL_NUM = float('1e-18')
CG,GC,AU,UA,GU,UG = range(6)
A,C,G,U = range(4)
PENALTY = 10.

def paired(c1, c2):
    _allowed_pairs = {"CG": -3, "GC": -3, "AU": -2, "UA":-2, "GU": -1, "UG":-1}
    c1, c2 = nucs[c1], nucs[c2]
    if c1 + c2 in _allowed_pairs:
        return _allowed_pairs[c1 + c2]
    else:
        return PENALTY

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

def projection_simplex_np_batch_coupled(params, z=1): # Tian Shuo's projection code
    for idx, x in params.items():
        x = np.array([x])
        x_sorted = np.sort(x, axis=1)[:, ::-1]
        cumsum = np.cumsum(x_sorted, axis=1)
        denom = np.arange(x.shape[1]) + 1
        theta = (cumsum - z)/denom
        mask = x_sorted > theta 
        csum_mask = np.cumsum(mask, axis=1)
        index = csum_mask[:, -1] - 1
        x_proj = np.maximum(x.transpose() - theta[np.arange(len(x)), index], 0)
        params[idx] = x_proj.transpose().flatten()

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
                    paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][nuci_1] + SMALL_NUM) + np.log(X[j][nucj] + SMALL_NUM) + (-paired(nuci_1, nucj) / RT))
                
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

    return np.exp(gradient)


def expected_free_energy(struct, X, coupled):
    n = len(struct)
    free_energy = 0.

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

            if not coupled:
                for nuci, nucj in _invalid_pairs:
                    score_ij += X[i][nuci] * X[j][nucj] * PENALTY
                    gradient[i][nuci] += X[j][nucj] * PENALTY
                    gradient[j][nucj] += X[i][nuci] * PENALTY

            free_energy += score_ij
        else:
            free_energy += unpaired()

    return free_energy, gradient

def marginalize(params, X):
    """
        Create n x 4 marginalized probability distribution
        X[i] = [p(A), p(C), p(G), p(U)]
    """
    for idx, prob in params.items():
        i, j = idx
        if i == j:
            X[j] = prob
        else:
            # 0-CG, 1-GC, 2-AU, 3-UA, 4-GU, 5-UG
            X[i] = np.array([prob[AU], prob[CG], prob[GC] + prob[GU], prob[UA] + prob[UG]])
            X[j] = np.array([prob[UA], prob[GC], prob[CG] + prob[UG], prob[AU] + prob[GU]])

def params_init(rna_struct, init_mode, coupled):
    """Return n x 6 matrix, each row is probability of CG, GC, AU, UA, GU, UG"""
    n = len(rna_struct)

    if not coupled:
        if init_mode == 'uniform':
            return np.array([[.25, .25, .25, .25] for _ in range(n)])

    params = {}
    stack = []

    for j, c in enumerate(rna_struct):
        if c == '(':
            stack.append(j)
        elif c == ')':
            i = stack.pop() # i, j paired

            if init_mode == 'uniform':
                # params[i, j] = {'CG': 1/6, 'GC': 1/6, 'AU': 1/6, 'UA': 1/6, 'GU': 1/6, 'UG': 1/6}
                params[i, j] = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
            elif init_mode == 'targeted':
                if np.random.randint(2):
                    params[i, j] = np.array([0.49, 0.51, 0., 0., 0., 0.])
                else:
                    params[i, j] = np.array([0.51, 0.49, 0., 0., 0., 0.])
        else:
            if init_mode == 'uniform' or init_mode == 'targeted': # j unpaired
                # params[j, j] = {'A': .25, 'C': .25, 'G': .25, 'U': .25}
                params[j, j] = np.array([.25, .25, .25, .25])
    
    return params

def get_intergral_solution(rna_struct, params, n, coupled):
    seq = ['A' for _ in range(n)]

    if coupled:
        for idx, prob in params.items():
            i, j = idx
            if i == j:
                seq[i] = nucs[np.argmax(prob)]
            else:
                seq[i] = nucpairs[np.argmax(prob)][0]
                seq[j] = nucpairs[np.argmax(prob)][1]
    else:
        stack = []
        for j, c in enumerate(rna_struct):
            if c == '(':
                stack.append(j)
            elif c == ')':
                i = stack.pop()
                prob = -1.

                for nuci, nucj in _allowed_pairs:
                    if params[i][nuci] * params[j][nucj] >= prob:
                        prob = params[i][nuci] * params[j][nucj]
                        seq[i] = nucs[nuci]
                        seq[j] = nucs[nucj]
            else:
                seq[j] = nucs[np.argmax(params[j])]

    return "".join(seq)

def gradient_descent(rna_struct, lr, num_step, sharpturn, init, coupled, results_file):
    total_start_time = time.time()
    n = len(rna_struct)
    log = []

    # starting distribution
    X = np.array([[.0, .0, .0, .0] for i in range(n)]) # n x 4 distribution
    if coupled:
        params = init # pair position is couplled
    else:
        X = init

    for epoch in range(num_step):
        start_time = time.time()

        curr_seq = ''
        if coupled:
            curr_seq = get_intergral_solution(rna_struct, params, n, coupled)
        else:
            curr_seq = get_intergral_solution(rna_struct, X, n, coupled)

        if coupled:
            marginalize(params, X)

        log_Q_hat = expected_inside_partition_log(X, sharpturn)
        grad1 = expected_outside_partition_log(log_Q_hat, X, sharpturn)

        # Delta_G, grad2 = expected_free_energy(rna_struct, params)
        Delta_G, grad2 = expected_free_energy(rna_struct, X, coupled)

        objective_value = Delta_G + log_Q_hat[n-1][0]
        
        # update step
        grad = grad1 + grad2

        if coupled:
            for idx, prob in params.items():
                i, j = idx
                if i == j:
                    params[i, j] -= lr * grad[i]
                else:
                    params[i, j] -= lr * np.array([grad[i][C] + grad[j][G],
                                                   grad[i][G] + grad[j][C],
                                                   grad[i][A] + grad[j][U],
                                                   grad[i][U] + grad[j][A],
                                                   grad[i][G] + grad[j][U],
                                                   grad[i][U] + grad[j][G]])

            projection_simplex_np_batch_coupled(params)
        else:
            X = X - (lr * grad)
            X = projection_simplex_np_batch(X)

        # if coupled:
        #     results_file.write('\n')
        #     for idx, prob in params.items():
        #         i, j = idx
        #         if i == j:
        #             results_file.write(f"{idx[0]} (unpaired), grad: {grad[i]}\n")
        #         else:
        #             results_file.write(f"{idx[0]} {idx[1]} (paired), grad: {np.array([grad[i][C] + grad[j][G],grad[i][G] + grad[j][C],grad[i][A] + grad[j][U],grad[i][U] + grad[j][A],grad[i][G] + grad[j][U],grad[i][U] + grad[j][G]])}\n")
        #     results_file.write('\n')

        end_time = time.time()
        elapsed_time = end_time - start_time

        results_file.write(f'step: {epoch+1: 5d}, objective value: {objective_value:.12f}, seq: {curr_seq}, time: {elapsed_time:.2f}\n')
        log.append(objective_value)

        if len(log) >= 2 and abs(log[-1] - log[-2]) < 1e-8:
            break

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    results_file.write("Final Distribution")
    if coupled:
        for idx, x in sorted(params.items()):
            i, j = idx
            if i == j:
                results_file.write(f"{i:2d}: A {x[A]:.2f}, C {x[C]:.2f}, G {x[G]:.2f}, U {x[U]:.2f}\n")
            else:
                results_file.write(f"({i}, {j}): CG {x[CG]:.2f} GC {x[GC]:.2f} AU {x[AU]:.2f} UA {x[UA]:.2f} GU {x[GU]:.2f} UG {x[UG]:.2f}\n")
    else:
        for i, x in enumerate(X):
            results_file.write(f"{i:2d}: A {x[A]:.2f}, C {x[C]:.2f}, G {x[G]:.2f}, U {x[U]:.2f}\n")
    
    if coupled:
        seq = get_intergral_solution(rna_struct, params, n, coupled)
    else:
        seq = get_intergral_solution(rna_struct, X, n, coupled)

    results_file.write(f"Optimal Sequence\n{rna_struct}\n{seq}\n")

    results_file.write(f"Total Elapsed Time\n{total_elapsed_time:.2f}\n")

    # plt.plot(log)
    # plt.plot_size(60, 20)
    # plt.xlabel('step')
    # plt.ylabel('Objective Value')
    # plt.title(f'{rna_struct}: n={n}, lr = {lr}, step={len(log)}')
    # plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--step", type=int, default=2000)
    parser.add_argument("--sharpturn", type=int, default=3)
    parser.add_argument("--init", type=str, default='uniform')
    parser.add_argument("--path", type=str, default='temp')
    parser.add_argument("--nocoupled", action='store_true', default=False)
    parser.add_argument("--test", action='store_true', default=False)
    args = parser.parse_args()
    print("Args: ", end='')
    print(args)

    # rna_struct = args.structure
    lr = args.lr
    num_step = args.step
    sharpturn = args.sharpturn
    init_mode = args.init
    result_folder = args.path
    coupled = not args.nocoupled
    test = args.test

    np.random.seed(seed=42)

    if test:
        if not os.path.exists(f'results/{result_folder}'):
            os.makedirs(f'results/{result_folder}')

        with open('short_eterna.txt', 'r') as data_file:
            lines = data_file.read().split('\n')
            lines = [line.split(' ') for line in lines]

            init = {}
            for line in lines:
                rna_id, rna_struct = line
                init[rna_id] = params_init(rna_struct, init_mode, coupled)

            def optimize(line):
                rna_id, rna_struct = line
                result_filepath = f'results/{result_folder}/{rna_id}.txt'

                with open(result_filepath, 'w+') as results_file:
                    results_file.write(f"{rna_struct}\n")
                    if coupled:
                        for idx, prob in init[rna_id].items():
                            results_file.write(f"{idx[0]} {idx[1]}, {prob}\n")
                    else:
                        for i, x in enumerate(init[rna_id]):
                            results_file.write(f"{i:2d}: A {x[A]:.2f}, C {x[C]:.2f}, G {x[G]:.2f}, U {x[U]:.2f}\n")
                    gradient_descent(rna_struct, lr, num_step, sharpturn, init[rna_id], coupled, results_file)

            num_threads = 8
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(optimize, line) for line in lines]
                concurrent.futures.wait(futures)

            print("All functions have completed.")
    else:
        rna_struct = input('Structure: ')
        init = params_init(rna_struct, init_mode, coupled)

        with open('result.txt', 'w+') as results_file:
            results_file.write(f"{rna_struct}\n")
            if coupled:
                for idx, prob in init.items():
                    results_file.write(f"{idx[0]} {idx[1]}, {prob}\n")
            else:
                for i, x in enumerate(init):
                    results_file.write(f"{i:2d}: A {x[A]:.2f}, C {x[C]:.2f}, G {x[G]:.2f}, U {x[U]:.2f}\n")
            gradient_descent(rna_struct, lr, num_step, sharpturn, init, coupled, results_file)
            


if __name__ == '__main__':
    main()

