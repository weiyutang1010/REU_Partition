import sys
import argparse
import numpy as np

import RNA
import concurrent.futures

sequences = []
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

    return np.log(value) * -kT / 100.0, value

# def jensen_approx(D):
#     # D = np.array([[.25,.25,.25,.25] for _ in range(n)]) # random distribution
#     # print(D)

#     with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
#         futures = [executor.submit(log_Q, "".join(seq)) for seq in sequences[n]]
#         concurrent.futures.wait(futures)
#     partition_jensen = log_E_Q(D)
#     print("Jensen value: ", partition_jensen)

def second_order_approx(D, E_Qx):
    """Brute Force, E_Qx = E[Q(x)]"""
    n = len(D)

    E_Qx_squared = 0. # E[Q(x)^2]
    for seq in sequences:
        Q_x = np.exp(log_Q_cached[seq] * 100.0 / -kT)
        E_Qx_squared += probability(D, seq) * (Q_x * Q_x)

    var_Qx = E_Qx_squared - (E_Qx * E_Qx)
    second_order = np.log(E_Qx) - (var_Qx / (2 * E_Qx * E_Qx))

    return second_order * -kT / 100.0

# def second_order_approx(n):
#     D = np.array([[.25,.25,.25,.25] for _ in range(n)]) # random distribution
#     print(D)

#     sequences[n] = generate_sequences('ACGU', n=n)
#     with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
#         futures = [executor.submit(log_Q, "".join(seq)) for seq in sequences[n]]
#         concurrent.futures.wait(futures)
#     partition_second_order = V_Q(D)
#     print("Second Order value: ", partition_second_order)

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

def sampling_test(n, k):
    # Given a distribution of length n
    D = np.array([[.25,.25,.25,.25] for _ in range(n)]) # random distribution
    print(D)

    # sequences[n] = generate_sequences('ACGU', n=n)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = [executor.submit(log_Q, "".join(seq)) for seq in sequences[n]]
    #     concurrent.futures.wait(futures)

    read_sequences(n)

    partition_jensen, E_Qx = log_E_Q(D)
    partition_exact = E_log_Q(D)
    partition_second = second_order_approx(D, E_Qx)

    print("Exact value (full): ", partition_exact)
    print("Jensen value (full): ", partition_jensen)
    print("Second Order Approximation (full): ", partition_second)
    # print("Second Order Approximation (full): ", 0.)

    samples = [np.random.choice(['A','C','G','U'], k, p=x) for x in D]
    samples = np.array(samples).transpose()
    seqs = ["".join(sample) for sample in samples]

    curr_sum = 0.
    for i, seq in enumerate(seqs):
        curr_sum += log_Q_cached[seq]
        print(i+1, curr_sum / (i + 1))

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
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--k", type=int, default=2000)
    args = parser.parse_args()

    for n in range(6, 9):
        with open(f'samples/n{n}.txt', 'w') as f:
            sys.stdout = f
            print(f"n: {n}, k: {args.k}")
            sampling_test(n, args.k)
