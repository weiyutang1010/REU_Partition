import sys
import argparse
import numpy as np

import RNA
import threading
import concurrent.futures

file_lock = threading.Lock()

def log_Q(x, file):
    fc = RNA.fold_compound(x)
    with file_lock:
        file.write(f"{x} {fc.pf()[1]:.20f}\n")

def generate_sequences(*args, n):
    pools = [tuple(pool) for pool in args] * n
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

for n in range(8, 13):
    with open(f'Qx/n{n}.txt', 'w') as file:
        sequences = generate_sequences('ACGU', n=n)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(log_Q, "".join(seq), file) for seq in sequences]
            concurrent.futures.wait(futures)

