import os
import sys
import time
import argparse
import json
import numpy as np
import math
from enum import Enum
from collections import defaultdict
import concurrent.futures

from utils import Mode, params_init, print_distribution, get_intergral_solution, marginalize
from objectives import objective

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
    """
        Works with coupled mode
    """
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

def gradient_descent(rna_id, rna_struct, dist, mode, results_file):
    total_start_time = time.time()
    n = len(rna_struct)
    log = []

        
    curr_seq =  get_intergral_solution(rna_struct, dist, n, mode)

    objective_value, grad, grad1, grad2 = objective(rna_struct, dist, mode)
    results_file.write(f'step: {0}, objective value: {objective_value:.12f}, seq: {curr_seq}, time: {0.:.2f}\n')
    print_distribution(dist, mode, results_file)

    for epoch in range(mode.num_steps):
        start_time = time.time()

        # if mode.coupled:
        #     marginalize(dist, X)
        # else:
        #     X = dist
        
        objective_value, grad, grad1, grad2 = objective(rna_struct, dist, mode)

        if mode.obj[-2:] == 'Dy':
            for idx, prob in dist.items():
                i, j = idx
                dist[i, j] -= mode.lr * grad[i, j]
                projection_simplex_np_batch_coupled(dist)
        elif mode.coupled:
            for idx, prob in dist.items():
                i, j = idx
                if i == j:
                    dist[i, j] -= mode.lr * grad[i]
                else:
                    dist[i, j] -= mode.lr * np.array([grad[i][C] + grad[j][G],
                                                   grad[i][G] + grad[j][C],
                                                   grad[i][A] + grad[j][U],
                                                   grad[i][U] + grad[j][A],
                                                   grad[i][G] + grad[j][U],
                                                   grad[i][U] + grad[j][G]])

            projection_simplex_np_batch_coupled(dist)
        else:
            X = X - (mode.lr * grad)
            X = projection_simplex_np_batch(X)
            dist = X

        # Debug: Print Gradient
        if mode.obj[-2:] == 'Dy':
            results_file.write('\nE[Delta G(D_y, y)] + E[log Q(D_y)] gradient\n')
            for idx, prob in sorted(dist.items()):
                i, j = idx
                if i == j:
                    results_file.write(f"{i}, grad: A {grad[i, j][A]:.4f} C {grad[i, j][C]:.4f} G {grad[i, j][G]:.4f} U {grad[i, j][U]:.4f}\n")
                else:
                    results_file.write(f"{i, j}, grad: CG {grad[i, j][CG]:.4f} CG {grad[i, j][GC]:.4f} AU {grad[i, j][AU]:.4f} UA {grad[i, j][UA]:.4f} GU {grad[i, j][GU]:.4f} UG {grad[i, j][UG]:.4f}\n")
            results_file.write('\n')

            grad = grad2
            results_file.write('E[Delta G(D_y, y)]gradient\n')
            for idx, prob in sorted(dist.items()):
                i, j = idx
                if i == j:
                    results_file.write(f"{i}, grad: A {grad[i, j][A]:.4f} C {grad[i, j][C]:.4f} G {grad[i, j][G]:.4f} U {grad[i, j][U]:.4f}\n")
                else:
                    results_file.write(f"{i, j}, grad: CG {grad[i, j][CG]:.4f} CG {grad[i, j][GC]:.4f} AU {grad[i, j][AU]:.4f} UA {grad[i, j][UA]:.4f} GU {grad[i, j][GU]:.4f} UG {grad[i, j][UG]:.4f}\n")
            results_file.write('\n')

            grad = grad1
            results_file.write('E[log Q(D_y)] gradient\n')
            for idx, prob in sorted(dist.items()):
                i, j = idx
                if i == j:
                    results_file.write(f"{i}, grad: A {grad[i, j][A]:.4f} C {grad[i, j][C]:.4f} G {grad[i, j][G]:.4f} U {grad[i, j][U]:.4f}\n")
                else:
                    results_file.write(f"{i, j}, grad: CG {grad[i, j][CG]:.4f} CG {grad[i, j][GC]:.4f} AU {grad[i, j][AU]:.4f} UA {grad[i, j][UA]:.4f} GU {grad[i, j][GU]:.4f} UG {grad[i, j][UG]:.4f}\n")
            results_file.write('\n')

        # if mode.coupled:
        #     mode.coupled = False
        #     # print_distribution(X, mode, results_file)
        #     mode.coupled = True
        #     results_file.write('\nE[Delta G(x, y)] + E[log Q(x)] gradient\n')

        #     for idx, prob in dist.items():
        #         i, j = idx
        #         if i == j:
        #             results_file.write(f"{idx[0]} (unpaired), grad: A {grad[i][A]:.4f} C {grad[i][C]:.4f} G {grad[i][G]:.4f} U {grad[i][U]:.4f}\n")
        #         else:
        #             results_file.write(f"{idx[0]} {idx[1]} (paired), grad: CG {grad[i][C] + grad[j][G]:.4f} CG {grad[i][G] + grad[j][C]:.4f} AU {grad[i][A] + grad[j][U]:.4f} UA {grad[i][U] + grad[j][A]:.4f} GU {grad[i][G] + grad[j][U]:.4f} UG {grad[i][U] + grad[j][G]:.4f}\n")
        #     results_file.write('\n')

            # grad = grad1
            # results_file.write('E[log Q(x)] gradient\n')
            # for idx, prob in dist.items():
            #     i, j = idx
            #     if i == j:
            #         results_file.write(f"{idx[0]} (unpaired), grad: A {grad[i][A]:.4f} C {grad[i][C]:.4f} G {grad[i][G]:.4f} U {grad[i][U]:.4f}\n")
            #     else:
            #         results_file.write(f"{idx[0]} {idx[1]} (paired), grad: CG {grad[i][C] + grad[j][G]:.4f} CG {grad[i][G] + grad[j][C]:.4f} AU {grad[i][A] + grad[j][U]:.4f} UA {grad[i][U] + grad[j][A]:.4f} GU {grad[i][G] + grad[j][U]:.4f} UG {grad[i][U] + grad[j][G]:.4f}\n")
            # results_file.write('\n')

            # grad = grad2
            # results_file.write('E[Delta G(x, y)] gradient\n')
            # for idx, prob in dist.items():
            #     i, j = idx
            #     if i == j:
            #         results_file.write(f"{idx[0]} (unpaired), grad: A {grad[i][A]:.4f} C {grad[i][C]:.4f} G {grad[i][G]:.4f} U {grad[i][U]:.4f}\n")
            #     else:
            #         results_file.write(f"{idx[0]} {idx[1]} (paired), grad: CG {grad[i][C] + grad[j][G]:.4f} CG {grad[i][G] + grad[j][C]:.4f} AU {grad[i][A] + grad[j][U]:.4f} UA {grad[i][U] + grad[j][A]:.4f} GU {grad[i][G] + grad[j][U]:.4f} UG {grad[i][U] + grad[j][G]:.4f}\n")
            # results_file.write('\n')

        end_time = time.time()
        elapsed_time = end_time - start_time

        curr_seq =  get_intergral_solution(rna_struct, dist, n, mode)
        results_file.write(f'step: {epoch+1}, objective value: {objective_value:.12f}, seq: {curr_seq}, time: {elapsed_time:.2f}\n')
        print(f'rna_id: {rna_id}, step: {epoch+1}, time: {elapsed_time:.2f}')
        print_distribution(dist, mode, results_file)
        log.append(objective_value)
        results_file.flush()

        if len(log) >= 2 and abs(log[-1] - log[-2]) < 1e-8:
            break

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    results_file.write("Final Distribution\n")
    print_distribution(dist, mode, results_file)

    seq =  get_intergral_solution(rna_struct, dist, n, mode)

    results_file.write(f"Optimal Sequence\n{rna_struct}\n{seq}\n")

    results_file.write(f"Total Elapsed Time\n{total_elapsed_time:.2f}\n")


def optimize(line, initial_distributions, results_folder, mode):
    rna_id, rna_struct = line
    result_filepath = f'results/{results_folder}/{rna_id}.txt'
    dist = initial_distributions[rna_id]

    with open(result_filepath, 'w+') as results_file:
        results_file.write(f'{rna_struct}\n')
        mode.print(results_file)
        gradient_descent(rna_id, rna_struct, dist, mode, results_file)

def main():
    # np.random.seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--step", type=int, default=2000)
    parser.add_argument("--sharpturn", type=int, default=3)
    parser.add_argument("--penalty", type=int, default=10)
    parser.add_argument("--init", type=str, default='uniform')
    parser.add_argument("--obj", type=str, default='pyx_jensen')
    parser.add_argument("--path", type=str, default='temp2')
    parser.add_argument("--nocoupled", action='store_true', default=False)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--energy", type=str, default='nussinov')
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--data", type=str, default='short_eterna.txt')
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    mode = Mode(args.lr, args.step, args.sharpturn, args.penalty, not args.nocoupled, args.k, args.energy, args.test, args.init, args.obj)
    result_folder = args.path

    if args.test:
        # Set up the results folder
        if not os.path.exists(f'data/{args.data}'):
            print("Error: Data File not found.")
            exit()

        if not os.path.exists(f'results/{result_folder}'):
            os.makedirs(f'results/{result_folder}')

        if not os.path.exists(f'results'):
            os.makedirs(f'results')

        # Read in structures from the data file
        lines, initial_distributions = [], {}

        with open(f'data/{args.data}', 'r') as data_file:
            lines = data_file.read().split('\n')
            lines = [line.split(' ') for line in lines]

        for line in lines:
            rna_id, rna_struct = line
            initial_distributions[rna_id] = params_init(rna_struct, mode)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(optimize, line, initial_distributions, result_folder, mode) for line in lines]
            concurrent.futures.wait(futures)

        print("All functions have completed.")
    else:
        rna_struct = input('Structure: ')
        line = [args.id, rna_struct]
        init = {args.id: params_init(rna_struct, mode)}
        optimize(line, init, 'temp', mode)
        print("Completed")

if __name__ == '__main__':
    main()

