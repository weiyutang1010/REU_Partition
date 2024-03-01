import sys
import argparse
import numpy as np

import RNA
import threading
import concurrent.futures

from collections import defaultdict

file_lock = threading.Lock()

NEG_INF = -1e18
_allowed_pairs = {"CG": -3, "GC": -3, "AU": -2, "UA":-2, "GU": -1, "UG":-1}

def valid(x, y):
    _allowed_pairs = {'CG', 'GC', 'AU', 'UA', 'GU', 'UG'}
    for i, j in [(0, 10), (1, 9), (2, 8), (3, 7)]:
    # for i, j in [(0, 8), (1, 7), (2, 6)]:
        if x[i] + x[j] not in _allowed_pairs:
            return False
    return True

nucs = 'ACGU'
def paired(c1, c2):
    if c1 + c2 in _allowed_pairs:
        return _allowed_pairs[c1 + c2]
    else:
        return 10

def unpaired(c='A'):
    return 1. # for now set every unpaired to same score

def log_Q(x, y, file):
    if not valid(x, y):
        return

    n = len(x)
    Q = defaultdict(lambda: defaultdict(lambda: NEG_INF))

    for j in range(n):
        Q[j-1][j] = 0.

    for j in range(n):
        for i in Q[j-1]:
            Q[j][i] = np.logaddexp(Q[j][i], Q[j-1][i] + (-unpaired(x[j])))

            if i > 0 and j - (i-1) > 3 and x[i-1] + x[j] in _allowed_pairs:
                for k in Q[i-2]:
                    Q[j][k] = np.logaddexp(Q[j][k], Q[i-2][k] + Q[j-1][i] + (-paired(x[i-1], x[j])))

    with file_lock:
        file.write(f"{x} {Q[n-1][0]:.10f}\n")

def generate_sequences(*args, n):
    pools = [tuple(pool) for pool in args] * n
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

# for n in range(9, 13):
#     with open(f'Qx/n{n}.txt', 'w') as file:
#         sequences = generate_sequences('ACGU', n=n)
#         with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
#             futures = [executor.submit(log_Q, "".join(seq), file) for seq in sequences]
#             concurrent.futures.wait(futures)

y = sys.argv[1]
n = len(y)

with open(f'Qx_nussinov/n{n}_y.txt', 'w') as file:
    sequences = generate_sequences('ACGU', n=n)
    print("Generation Done")

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(log_Q, "".join(seq), y, file) for seq in sequences]
        concurrent.futures.wait(futures)


