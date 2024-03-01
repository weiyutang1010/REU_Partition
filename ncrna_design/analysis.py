import os, sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import generate_sequences

def paired(c1, c2):
    _allowed_pairs = {"CG": -3, "GC": -3, "AU": -2, "UA":-2, "GU": -1, "UG":-1}
    if c1 + c2 in _allowed_pairs:
        return _allowed_pairs[c1 + c2]
    else:
        return 10

def unpaired(c='A'):
    return 1. # for now set every unpaired to same score

def free_energy(rna_seq, rna_struct):
    n = len(rna_seq)
    stack = []
    score = 0.

    for j, c in enumerate(rna_struct):
        if c == '(':
            stack.append(j)
        elif c == ')':
            i = stack.pop()
            score += paired(rna_seq[i], rna_seq[j])
        else:
            score += unpaired(c)

    return score

def inside_prob(x, sharpturn=0):
    """Left to Right"""
    _allowed_pairs = {'CG', 'GC', 'AU', 'UA', 'GU', 'UG'}

    n = len(x)
    Q = [defaultdict(float) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 1.

    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] += Q[j-1][i] * np.exp(-unpaired(x[j-1]))

            if i > 1 and j - (i-1) > sharpturn and x[i-2] + x[j-1] in _allowed_pairs:
                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i] * np.exp(-paired(x[i-2], x[j-1]))

    return Q

def get_boltz_prob(rna_seq, rna_struct, sharpturn=0):
    n = len(rna_seq)
    Q = inside_prob(rna_seq, sharpturn)
    delta_G =  free_energy(rna_seq, rna_struct)

    return np.exp(-delta_G) / Q[n][1]

def process_result_file(rna_id, result_file, baseline_file):
    lines = result_file.read().split('\n')
    b_lines = baseline_file.read().split('\n') # b for baseline (comparison)
    n, rna_struct = len(lines[0]), lines[0]

    obj, b_obj = [], []
    seqs, b_seqs = [], []
    for line in lines:
        if line.startswith("step: "):
            obj.append(float(line.split(', ')[1].split(': ')[1]))
            seqs.append(line.split(', ')[2].split(': ')[1])

    for line in b_lines:
        if line.startswith("step: "):
            b_obj.append(float(line.split(', ')[1].split(': ')[1]))
            b_seqs.append(line.split(', ')[2].split(': ')[1])

    seq_idx = lines.index('Optimal Sequence') + 2
    print(f"id: {rna_id}")
    print(rna_struct)
    print(lines[seq_idx])

    test = get_boltz_prob(seqs[0], rna_struct)

    # fractional solutions
    initial, final = np.exp(-obj[0]), np.exp(-obj[-1])
    b_final = np.exp(-b_obj[-1])

    print("Fractional Solution")
    print(f"Initial: {initial:.5f}, Final: {final:5f}, Diff: {final-initial:.5f} (exp[inf p(y | x)])")
    print(f"Coupled: {final:5f}, No Coupled: {b_final:5f}, Diff: {final-b_final:.5f} (exp[inf p(y | x)])")


    # integral solutions
    initial, final = get_boltz_prob(seqs[0], rna_struct), get_boltz_prob(seqs[-1], rna_struct)
    b_final = get_boltz_prob(b_seqs[-1], rna_struct)

    print("Integral Solution")
    print(f"Initial: {initial:.5f}, Final: {final:5f}, Diff: {final-initial:.5f} (p(y | x))")
    print(f"Coupled: {final:5f}, No Coupled: {b_final:5f}, Diff: {final-b_final:.5f} (p(y | x))")
    print()

def process_result_file(rna_id, result_file):
    lines = result_file.read().split('\n')
    n, rna_struct = len(lines[0]), lines[0]

    sharpturn = int(lines[1].split(', ')[2].split(': ')[1])
    lr = float(lines[1].split(', ')[0].split(': ')[1])

    objs, seqs, pyx = [], [], []
    for line in lines:
        if line.startswith("step:"):
            objs.append(float(line.split(', ')[1].split(': ')[1]))
            seqs.append(line.split(', ')[2].split(': ')[1])

    best_seq, best_score = '', 0.
    prev_seq, prev_score = '', 0.
    for idx, seq in enumerate(seqs):
        if seq != prev_seq:
            prev_seq = seq
            prev_score = get_boltz_prob(seq, rna_struct, sharpturn)
            pyx.append(prev_score)
            print(f"{idx}, {seq}, {prev_score:.6f}")
            if prev_score > best_score:
                best_seq = prev_seq
                best_score = prev_score
        else:
            pyx.append(prev_score)

    print(best_seq, best_score, len(np.unique(seqs)))

    objs_exp = [np.exp(-1 * obj) for obj in objs]

    plt.rcParams["figure.figsize"] = [7.50, 4.50]
    plt.rcParams["figure.autolayout"] = True

    fig, ax1 = plt.subplots()
    # plt.axvline(x=48, color='r', linestyle='--')

    ax1.set_xlabel('Step')
    # ax1.set_ylabel(r'$\mathbb{E}[\Delta G(x, y)]$', color='red')
    ax1.set_ylabel('Conditional Probability')
    # ax1.plot(objs, color='red', alpha=0.7, label=r'$\mathbb{E}[\Delta G(x, y)]$ expected free energy (simple)')
    ax1.plot(objs_exp, color='blue', alpha=0.6, label=r'Fractional $\exp \mathbb{E}[\log p(y|x)]$')
    ax1.plot(pyx, color='orange', alpha=0.6, label=r'Integral $p(y \mid x)$')
    # ax1.tick_params(axis='y', labelcolor='red')
    ax1.tick_params(axis='y')
    ax1.legend(fontsize="8")

    plt.title(f'id {rna_id}, Puzzle {rna_struct}, lr={lr}')
    plt.savefig(f'graphs/puzzle_{rna_id}.png', format="png", bbox_inches="tight")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--y", type=str, default="")
    parser.add_argument("--folder", type=str, default="temp")
    parser.add_argument("--file", type=str, default="0")

    parser.add_argument("--mode", type=str, default="best")
    args = parser.parse_args()

    # find best solutions
    if args.mode == "best":
        arr = []
        sol = {}

        rna_struct = args.y
        n = len(rna_struct)

        with open(f'./Qx_nussinov/n{n}_y.txt', 'r') as file:
            lines = file.read().split('\n')

            for line in lines:
                if len(line) > 0:
                    seq = line.split(' ')[0]
                    log_Q = float(line.split(' ')[1])
                    delta_G = free_energy(seq, rna_struct)
                    arr.append((-(np.exp(-delta_G) / np.exp(log_Q)), seq))
        arr.sort()

        prev_score = 0.
        for idx, x in enumerate(arr[:200]):
            if -x[0] != prev_score:
                print(f"{idx}, {x[1]} {-x[0]:.3f}")
                prev_score = -x[0]
    # graph
    elif args.mode == "graph":
        results_path = f'./results/{args.folder}/{args.file}.txt'
        with open(results_path, 'r') as result_file:
            process_result_file(args.file, result_file)

    # baseline_path = ''
    # if len(sys.argv) > 2:
    #     baseline_path = f'./results/{sys.argv[2]}'
    
    # ids = []
    # if os.path.exists(results_path):
    #     for folder_path, _, filenames in os.walk(results_path):
    #         for filename in filenames:
    #             rna_id, extension = os.path.splitext(filename)
    #             ids.append(int(rna_id))
        
    #     sort_by_len = [8, 1, 23, 26, 15, 30, 88, 41, 3, 11, 57, 66, 40, 65, 20, 10, 33, 47]
    #     # for rna_id in ids:
    #     for rna_id in sort_by_len:
    #         results_filepath = os.path.join(results_path, f'{rna_id}.txt')
    #         baseline_filepath = os.path.join(baseline_path, f'{rna_id}.txt')

    #         with open(results_filepath, 'r') as result_file:
    #             with open(baseline_filepath, 'r') as baseline_file:
    #                 process_result_file(rna_id, result_file, baseline_file)
    # else:
    #     print('Folder does not exist.')


if __name__ == '__main__':
    main()