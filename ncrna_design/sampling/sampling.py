import sys
import argparse
import numpy as np

import RNA
import concurrent.futures

sequences = {}
log_Q_cached = {}

def generate_sequences(*args, n):
    pools = [tuple(pool) for pool in args] * n
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

def log_Q(x):
    fc = RNA.fold_compound(x)
    log_Q_cached[x] = fc.pf()[1]

def probability(D, x):
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    p = 1.
    for idx, c in enumerate(x):
        p *= D[idx][nuc_to_idx[c]]
    return p


def E_log_Q(D):
    """Brute Force"""
    n = len(D)

    if n not in sequences:
        sequences[n] = generate_sequences('ACGU', n=n)

    value = 0.
    for seq in sequences[n]:
        value += probability(D, seq) * log_Q_cached["".join(seq)]

    return value

def compare(n, k):
    # Given a distribution of length n
    D = np.random.dirichlet(np.ones(4), size=n) # random distribution

    # Calculate entropy
    entropy = 0.
    for seq in sequences[n]:
        prob = probability(D, seq)
        if prob > 0:
            entropy -= prob * np.log2(prob)

    # Sampling
    samples = []
    for x in D:
        samples.append(np.random.choice(['A','C','G','U'], k, p=x))
    samples = np.array(samples).transpose()
    
    seqs = ["".join(sample) for sample in samples]

    partition_sampled = np.sum([log_Q_cached[seq] for seq in seqs]) / k
    partition_exact = E_log_Q(D)

    print(f"{entropy}, {partition_exact - partition_sampled}")
    sys.stdout.flush()

def approximation_gap(n, k, t):
    # length - n, sample size - k
    print("Entropy, Approximation Gap")

    sequences[n] = generate_sequences('ACGU', n=n)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(log_Q, "".join(seq)) for seq in sequences[n]]
        concurrent.futures.wait(futures)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # Define tasks to be executed concurrently
        futures = [executor.submit(compare, n, k) for _ in range(t)]
        concurrent.futures.wait(futures)
    
def sampling_test(n, k):
    # Given a distribution of length n
    # D = np.random.dirichlet(np.ones(4), size=n) # random distribution
    D = np.array([[.25,.25,.25,.25] for _ in range(n)]) # random distribution
    print(D)

    sequences[n] = generate_sequences('ACGU', n=n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(log_Q, "".join(seq)) for seq in sequences[n]]
        concurrent.futures.wait(futures)
    partition_exact = E_log_Q(D)
    print("Exact value: ", partition_exact)

    samples = []
    for x in D:
        samples.append(np.random.choice(['A','C','G','U'], k, p=x))
    samples = np.array(samples).transpose()
    seqs = ["".join(sample) for sample in samples]

    curr_sum = 0.
    for i, seq in enumerate(seqs):
        curr_sum += log_Q_cached[seq]
        print(i+1, curr_sum / (i + 1))
    
if __name__ == '__main__':
    np.random.seed(seed=42)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--step", type=int, default=2000)
    parser.add_argument("--sharpturn", type=int, default=3)
    parser.add_argument("--penalty", type=int, default=10)
    parser.add_argument("--obj", type=str, default='pyx_jensen')
    parser.add_argument("--path", type=str, default='temp2')
    parser.add_argument("--nocoupled", action='store_true', default=False)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--data", type=str, default='short_eterna.txt')
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--init", type=str, default='uniform')

    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--k", type=int, default=10000)
    parser.add_argument("--t", type=int, default=1000)
    args = parser.parse_args()

    print(f"n: {args.n}, k: {args.k}, t: {args.t}")
    # test(args.n, args.k, args.t)
    sampling_test(args.n, args.k)
