import sys
import argparse
import numpy as np

import RNA
import threading
import concurrent.futures

sequences = []
log_Q_cached = {}

file_lock = threading.Lock()

# def generate_sequences(*args, n):
#     pools = [tuple(pool) for pool in args] * n
#     result = [[]]
#     for pool in pools:
#         result = [x+[y] for x in result for y in pool]
#     return result

def generate_distribution(n, y):
    stack = []
    paired, unpaired = [], []
    for j, c in enumerate(y):
        if c == '(':
            stack.append(j)
        elif c == ')':
            i = stack.pop()
            paired.append((i, j))
        else:
            unpaired.append(j)

    test_distribution = {}

    for j in unpaired:
        rand_int = np.random.randint(1, 5)
        arr = np.random.dirichlet(np.ones(rand_int))
        arr = np.pad(arr, (0, 4 - len(arr)))
        np.random.shuffle(arr)
        test_distribution[j, j] = arr

    for i, j in paired:
        rand_int = np.random.randint(1, 7)
        arr = np.random.dirichlet(np.ones(rand_int))
        arr = np.pad(arr, (0, 6 - len(arr)))
        np.random.shuffle(arr)
        test_distribution[i, j] = arr

    return test_distribution

def log_Q(x):
    fc = RNA.fold_compound(x)
    log_Q_cached[x] = fc.pf()[1]

def probability(D, x):
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    nucpair_to_idx = {'CG': 0, 'GC': 1, 'AU': 2, 'UA': 3, 'GU': 4, 'UG': 5}

    p =  1.
    for i, j in D:
        if i == j:
            p *= D[j, j][nuc_to_idx[x[j]]]
        else:
            p *= D[i, j][nucpair_to_idx[x[i] + x[j]]]

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

def compare(n, k, y, D=None):
    n = len(y)

    # Given a distribution of length n
    if not D:
        D = generate_distribution(n, y) # random distribution

    # Calculate entropy
    entropy = 0.
    for idx, x in D.items():
        for x_i in x:
            if x_i > 0.:
                entropy -= x_i * np.log2(x_i)
    entropy /= len(D)

    # Sampling
    stack = []
    paired, unpaired = [], []
    for j, c in enumerate(y):
        if c == '(':
            stack.append(j)
        elif c == ')':
            i = stack.pop()
            paired.append((i, j))
        else:
            unpaired.append(j)

    samples = []
    for _ in range(k):
        seq = ['A'] * len(y)
        for j in unpaired:
            seq[j] = np.random.choice(['A','C','G','U'], p=D[j,j])
        for i, j in paired:
            chosen = np.random.choice(['CG','GC','AU','UA', 'GU', 'UG'], p=D[i, j])
            seq[i] = chosen[0]
            seq[j] = chosen[1]
        samples.append("".join(seq))

    partition_sampled = np.sum([log_Q_cached[seq] for seq in samples]) / k
    partition_exact = E_log_Q(D)
    partition_jensen = log_E_Q(D)

    print(f"{entropy}, {partition_sampled}, {partition_jensen}, {partition_exact}")
    sys.stdout.flush()

def read_sequences(n):
    global sequences
    sequences = []
    global log_Q_cached
    log_Q_cached = {}

    with open(f'Qx/n{n}_y.txt', 'r') as file:
        lines = file.read().split('\n')
        for line in lines[:-1]:
            seq = line.split(' ')[0]
            sequences.append(seq)
            log_Q_cached[seq] = float(line.split(' ')[1])

def rand_one(n):
    arr = [0.] * n
    random_index = np.random.randint(0, n)
    arr[random_index] = 1.
    return arr

def approximation_gap(n, k, t, y):
    # length - n, sample size - k
    print("Entropy, Sampling, Jensen, Exact")

    read_sequences(n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # Define tasks to be executed concurrently
        futures = [executor.submit(compare, n, k, y) for _ in range(t)]
        concurrent.futures.wait(futures)

    stack = []
    paired, unpaired = [], []
    for j, c in enumerate(y):
        if c == '(':
            stack.append(j)
        elif c == ')':
            i = stack.pop()
            paired.append((i, j))
        else:
            unpaired.append(j)

    D = {}
    for j in unpaired:
        D[j, j] = np.array([.25, .25, .25, .25])
    for i, j in paired:
        D[i, j] = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
    compare(n, k , y, D)


    D = {}
    for j in unpaired:
        D[j, j] = np.array(rand_one(4))
    for i, j in paired:
        D[i, j] = np.array(rand_one(6))
    compare(n, k , y, D)


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
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--t", type=int, default=500)
    args = parser.parse_args()

    original_stdout = sys.stdout
    y = "((((...))))."
    n = len(y)

    with open(f'approx_gap_y/n{n}.txt', 'w') as f:
        sys.stdout = f
        print(f"n: {n}, k: {args.k}, t: {args.t}, y: {y}")
        approximation_gap(n, args.k, args.t, y)


    
