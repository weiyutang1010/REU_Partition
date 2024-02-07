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

def generate_distribution(n):
    test_distribution = []

    rand_int = np.random.randint(2, 5)

    for _ in range(n):

        if rand_int == 2:
            rand = np.concatenate((np.random.sample(2), np.array([1., 0.])), axis=0)
            rand.sort()

            test_distribution.append([
                rand[1] - rand[0],
                rand[2] - rand[1],
                0.,
                0.,
            ])
            np.random.shuffle(test_distribution[-1])

        if rand_int == 3:
            rand = np.concatenate((np.random.sample(2), np.array([1., 0.])), axis=0)
            rand.sort()

            test_distribution.append([
                rand[1] - rand[0],
                rand[2] - rand[1],
                rand[3] - rand[2],
                0.,
            ])
            np.random.shuffle(test_distribution[-1])

        if rand_int == 4:
            rand = np.concatenate((np.random.sample(3), np.array([1., 0.])), axis=0)
            rand.sort()

            test_distribution.append([
                rand[1] - rand[0],
                rand[2] - rand[1],
                rand[3] - rand[2],
                rand[4] - rand[3]
            ])

    return np.array(test_distribution)

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
    D = generate_distribution(n) # random distribution

    # Calculate entropy
    entropy = 0.
    for x in D:
        for x_i in x:
            if x_i > 0.:
                entropy -= x_i * np.log2(x_i)
    entropy /= n

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

kT = 61.63207755
def log_E_Q(D):
    """Brute Force"""
    n = len(D)

    if n not in sequences:
        sequences[n] = generate_sequences('ACGU', n=n)

    value = 0.
    for seq in sequences[n]:
        value += probability(D, seq) * np.exp(log_Q_cached["".join(seq)] * 100 / -kT)

    return value

def jensen_approx(n):
    D = np.array([[.25,.25,.25,.25] for _ in range(n)]) # random distribution
    print(D)

    sequences[n] = generate_sequences('ACGU', n=n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(log_Q, "".join(seq)) for seq in sequences[n]]
        concurrent.futures.wait(futures)
    partition_jensen = log_E_Q(D)
    print("Jensen value: ", np.log(partition_jensen) * -kT / 100.0)
    
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

    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--k", type=int, default=2000)
    parser.add_argument("--t", type=int, default=500)
    args = parser.parse_args()

    print(f"n: {args.n}, k: {args.k}, t: {args.t}")

    if args.mode == 1:
        approximation_gap(args.n, args.k, args.t)
    else:
        sampling_test(args.n, args.k)
