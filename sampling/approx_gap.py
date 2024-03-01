import sys
import argparse
import numpy as np

import RNA
import threading
import concurrent.futures

sequences = []
log_Q_cached = {}

file_lock = threading.Lock()

def generate_sequences(*args, n):
    pools = [tuple(pool) for pool in args] * n
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

def generate_distribution(t, n):
    T = np.linspace(0.05, 3, num=t) #temperature
    D = []

    for kT in T:
        dist = []
        for _ in range(n):
            rand_nums = np.random.rand(4)
            logits = np.log(rand_nums / (1 - rand_nums))
            scaled_logits = np.exp(logits / kT)
            Q = np.sum(scaled_logits)
            dist.append(scaled_logits / Q)
        D.append(np.array(dist))

    return D

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

    value = 0.
    for seq in sequences:
        value += probability(D, seq) * log_Q_cached[seq]

    return value

kT = 61.63207755
def log_E_Q(D):
    """Brute Force"""
    n = len(D)

    value = 0.
    for seq in sequences:
        value += probability(D, seq) * np.exp(log_Q_cached[seq] * 100.0 / -kT)

    return np.log(value) * -kT / 100.0

def compare(n, k, D=None):
    # Given a distribution of length n
    if D is None:
        D = generate_distribution(1, n)[0] # random distribution

    # Calculate entropy
    entropy = 0.
    for x in D:
        for x_i in x:
            if x_i > 0.:
                entropy -= x_i * np.log2(x_i)
    entropy /= n

    # Sampling
    samples = np.array([np.random.choice(['A','C','G','U'], k, p=x) for x in D]).transpose()
    
    seqs = ["".join(sample) for sample in samples]

    partition_sampled = np.sum([log_Q_cached[seq] for seq in seqs]) / k
    partition_exact = E_log_Q(D)
    partition_jensen = log_E_Q(D)

    print(f"{entropy}, {partition_sampled}, {partition_jensen}, {partition_exact}")
    sys.stdout.flush()


def read_sequences(n):
    global sequences
    sequences = []
    global log_Q_cached
    log_Q_cached = {}

    with open(f'Qx/n{n}.txt', 'r') as file:
        lines = file.read().split('\n')
        for line in lines[:-1]:
            seq = line.split(' ')[0]
            sequences.append(seq)
            log_Q_cached[seq] = float(line.split(' ')[1])

def approximation_gap(n, k, t):
    # length - n, sample size - k
    print("Entropy, Sampling, Jensen, Exact")

    D = generate_distribution(t, n) # generate t random distributions

    read_sequences(n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # Define tasks to be executed concurrently
        futures = [executor.submit(compare, n, k, d) for d in D]
        concurrent.futures.wait(futures)


if __name__ == '__main__':
    np.random.seed(seed=42)
    parser = argparse.ArgumentParser()

    # parser.add_argument("--id", type=int, default=0)
    # parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--step", type=int, default=2000)
    # parser.add_argument("--sharpturn", type=int, default=3)
    # parser.add_argument("--penalty", type=int, default=10)
    # parser.add_argument("--obj", type=str, default='pyx_jensen')
    # parser.add_argument("--path", type=str, default='temp2')
    # parser.add_argument("--nocoupled", action='store_true', default=False)
    # parser.add_argument("--test", action='store_true', default=False)
    # parser.add_argument("--data", type=str, default='short_eterna.txt')
    # parser.add_argument("--threads", type=int, default=8)
    # parser.add_argument("--init", type=str, default='uniform')

    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--t", type=int, default=700)
    args = parser.parse_args()

    original_stdout = sys.stdout
    for n in range(7, 11):
        with open(f'approx_gap/n{n}.txt', 'w') as f:
            sys.stdout = f
            print(f"n: {n}, k: {args.k}, t: {args.t}")
            approximation_gap(n, args.k, args.t)
        sys.stdout = original_stdout
        print(f"n = {n} done")


    
