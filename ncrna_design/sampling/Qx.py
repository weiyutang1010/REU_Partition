import sys
import argparse
import numpy as np

import RNA
import threading
import concurrent.futures

file_lock = threading.Lock()

def valid(x, y):
    _allowed_pairs = {'CG', 'GC', 'AU', 'UA', 'GU', 'UG'}
    # for i, j in [(0, 10), (1, 9), (2, 8), (3, 7)]:
    for i, j in [(0, 8), (1, 7), (2, 6)]:
        if x[i] + x[j] not in _allowed_pairs:
            return False
    return True

def log_Q(x, y, file):
    if not valid(x, y):
        return

    fc = RNA.fold_compound(x)
    with file_lock:
        file.write(f"{x} {fc.pf()[1]:.30f}\n")

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

n = 9
y = "(((...)))"

with open(f'Qx/n{n}_y.txt', 'w') as file:
    sequences = generate_sequences('ACGU', n=n)
    print("Generation Done")
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(log_Q, "".join(seq), y, file) for seq in sequences]
        concurrent.futures.wait(futures)


