import sys, os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import RNA

import random
random.seed(42)

sequences = []
seqs = []
probs = []

# Generate k samples without replacement
def p(D, x):
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    p = 1.
    for idx, c in enumerate(x):
        p *= D[idx][nuc_to_idx[c]]
    return p

def generate_sequences(*args, n):
    pools = [tuple(pool) for pool in args] * n
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

def next_seq(seq):
    next_char = {'A': 'C', 'C': 'G', 'G': 'U', 'U': 'A'}
    n = len(seq)
    
    idx = n - 1
    seq[idx] = next_char[seq[idx]]
    while idx > 0 and seq[idx] == 'A':
        idx -= 1
        seq[idx] = next_char[seq[idx]]


def main(n, sample_sizes, t):
    # Given a distribution of length n
    D = np.random.dirichlet(np.ones(3), size=n) # [[A, C, G, U]]
    zeros = np.zeros((n, 1))
    D = np.hstack((D, zeros))

    seq = ['A'] * n
    while len(seq) > 0:
        p_x = p(D, seq)
        if p_x > 0.0:
            seqs.append("".join(seq))
            probs.append(p(D, seq))
        seq = next_seq(seq)

    # Calculate the sum of p(x) * log Q(x) over each sample
    def log_Q(x):
        fc = RNA.fold_compound(x)
        return fc.pf()[1]
    
    saved_logQ = {}
    full = 0.
    for seq, prob in zip(seqs, probs):
        val = log_Q(seq)
        full += prob * val
        saved_logQ[seq] = val
    
    x_values = []
    y_values = []

    print(f"max sample = {len(seqs)}")
    for k in sample_sizes:
        print(f"n = {n}, k = {k}")
        avg_diff = 0.
        for trial in range(t):
            samples = np.random.choice(range(len(seqs)), min(len(seqs), k), p=probs, replace=False)
            
            approx = 0.
            for idx in samples:
                approx += probs[idx] * saved_logQ[seqs[idx]]
            avg_diff += abs(full - approx)
            # print(f"Full: {full:.5f}, Approximate: {approx:.5f}, Abs Diff = {full - approx:.5f}")   
            x_values.append(k)
            y_values.append(approx) 
        
        print(f"Full: {full:.5f}, Avg Diff: {abs(avg_diff / trial): .5f}")
    
    # Create scatterplot
    plt.scatter(x_values, y_values, color='blue', marker='x', label='Approx value')
    plt.axhline(y=full, color='red', linestyle='--', label='Exact value')

    # Add labels and title
    plt.xscale('log')
    plt.xlabel('Sample Size')
    plt.ylabel(r'$\mathbb{E}[\log Q(x)]$ (kcal/mol)')
    plt.title(fr'Approximated $\mathbb{{E}}[\log Q(x)]$, n = {n}, max_sample = {len(seqs)}')

    plt.grid(True)
    plt.legend()
    plt.savefig(f"sampling_result/{n}.png")

if __name__ == '__main__':
    main(int(sys.argv[1]), [100, 1000, 10000, 100000], 30)