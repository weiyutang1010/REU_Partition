import sys
import argparse
import numpy as np
import scipy as sc

import RNA

from utils import Mode, generate_sequences
import concurrent.futures
# from objectives import expected_inside_partition_log

sequences = {}

# def expected_inside_partition_log(X, mode):
#     """
#         log E[Q(x)]
#     """
#     n = len(X)
#     Q_hat = defaultdict(lambda: defaultdict(lambda: NEG_INF))

#     for j in range(n):
#         Q_hat[j-1][j] = 0.

#     for j in range(n):
#         for i in Q_hat[j-1]:
#             unpaired_sc = NEG_INF
#             for nucj in range(4):
#                 unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j][nucj] + SMALL_NUM) + (-unpaired(nucj) / RT))
#             Q_hat[j][i] = np.logaddexp(Q_hat[j][i], Q_hat[j-1][i] + unpaired_sc)

#             if i > 0 and j-(i-1) > mode.sharpturn:
#                 paired_sc = NEG_INF # calculate weighted paired score
#                 for nuci_1, nucj in _allowed_pairs:
#                     paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][nuci_1] + SMALL_NUM) + np.log(X[j][nucj] + SMALL_NUM) + (-paired(nuci_1, nucj, mode) / RT))
                
#                 for k in Q_hat[i-2]:
#                     Q_hat[j][k] = np.logaddexp(Q_hat[j][k], Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc)

#     return Q_hat

def log_Q(x):
    fc = RNA.fold_compound(x)
    return fc.pf()[1]

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
        value += probability(D, seq) * log_Q("".join(seq))

    return value

def compare(n, k):
    # Given a distribution of length n
    D = np.random.dirichlet(np.ones(4), size=n) # random distribution

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

    partition_sampled = np.sum(np.vectorize(log_Q)(seqs)) / k
    partition_exact = E_log_Q(D)

    print(f"{entropy}, {partition_exact - partition_sampled}")
    sys.stdout.flush()


def test(n, k, t, mode):
    # length - n, sample size - k
    print("Entropy, Approximation Gap")

    sequences[n] = generate_sequences('ACGU', n=n)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Define tasks to be executed concurrently
        futures = [executor.submit(compare, n, k) for _ in range(t)]
        concurrent.futures.wait(futures)
    
    # for _ in range(t):
    #     # Given a distribution of length n
    #     D = np.random.dirichlet(np.ones(4), size=n) # random distribution

    #     entropy = 0.

    #     if n not in sequences:
    #         sequences[n] = generate_sequences('ACGU', n=n)

    #     for seq in sequences[n]:
    #         prob = probability(D, seq)
    #         if prob > 0:
    #             entropy -= prob * np.log2(prob)

    #     # Sampling
    #     samples = []
    #     for x in D:
    #         samples.append(np.random.choice(['A','C','G','U'], k, p=x))
    #     samples = np.array(samples).transpose()
    #     seqs = ["".join(sample) for sample in samples]

    #     partition_sampled = np.sum(np.vectorize(log_Q)(seqs)) / k
    #     partition_exact = E_log_Q(D)

    #     print(f"{entropy}, {partition_exact - partition_sampled}")
    
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
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--t", type=int, default=1000)
    args = parser.parse_args()

    mode = Mode(args.lr, args.step, args.sharpturn, args.penalty, not args.nocoupled, args.test, args.init, args.obj)
    test(args.n, args.k, args.t, mode)

