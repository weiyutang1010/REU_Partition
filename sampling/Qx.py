import sys
import argparse
import numpy as np

import RNA
import threading
import concurrent.futures

import subprocess

from itertools import product

file_lock = threading.Lock()
LP_PATH = "LinearPartition/linearpartition"

def valid(x, y):
    _allowed_pairs = {'CG', 'GC', 'AU', 'UA', 'GU', 'UG'}
    stack = []
    for j, c in enumerate(y):
        if c == '(':
            stack.append(j)
        elif c == ')':
            i = stack.pop()
            if x[i] + x[j] not in _allowed_pairs:
                return False
    return True

def valid_target(x, y):
    _allowed_pairs = {'CG', 'GC'}
    stack = []
    for j, c in enumerate(y):
        if c == '(':
            stack.append(j)
        elif c == ')':
            i = stack.pop()
            if x[i] + x[j] not in _allowed_pairs:
                return False
        elif c == '.' and x[j] != 'A':
            return False
    return True

def valid_seq(y, targeted):
    x = []

    pairs = []
    unpairs = []
    stack = []
    for j, c in enumerate(y):
        if c == '(':
            stack.append(j)
        elif c == ')':
            pairs.append((stack.pop(), j))
        else:
            unpairs.append(j)

    if targeted:
        seq = ['A'] * len(y)
        for combination in product(['CG', 'GC'], repeat=len(pairs)):
            for idx, pair in enumerate(combination):
                i, j = pairs[idx]
                seq[i] = pair[0]
                seq[j] = pair[1]
            x.append("".join(seq))
    else:
        seq = ['A'] * len(y)
        for comb1 in product(['CG', 'GC', 'AU', 'UA', 'GU', 'UG'], repeat=len(pairs)):
            for comb2 in product(['A', 'C', 'G', 'U'], repeat=len(unpairs)):
                for idx, pair in enumerate(comb1):
                    i, j = pairs[idx]
                    seq[i] = pair[0]
                    seq[j] = pair[1]
                
                for idx, c in enumerate(comb2):
                    seq[unpairs[idx]] = c

                x.append("".join(seq))

    return x

def log_Q(x, file):
    cmds = f"./LinearPartition/linearpartition -V"
    rt = subprocess.run(cmds.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, input=f"{x}\n".encode())
    lines = rt.stderr.decode('utf-8').strip().split('\n')

    with file_lock:
        file.write(f"{x} {lines[0][25:33]}\n")

parser = argparse.ArgumentParser()
parser.add_argument("--y", type=str, default="")
parser.add_argument("--n", type=int, default=0)
parser.add_argument("--targeted", action='store_true', default=False)

args = parser.parse_args()

n = args.n
y = ""

if args.y != "":
    y = args.y
    n = len(y)

if y == "" and n > 0:
    with open(f'Qx/n{n}.txt', 'w') as file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(log_Q, "".join(seq), file) for seq in product('ACGU', repeat=n)]
            concurrent.futures.wait(futures)
elif y != "" and not args.targeted:
    with open(f'Qx/n{n}_y.txt', 'w') as file:
        file.write(f"{y}\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(log_Q, "".join(seq), file) for seq in valid_seq(y, args.targeted)]
            concurrent.futures.wait(futures)
elif y != "" and args.targeted:
    with open(f'Qx/n{n}_y_target.txt', 'w') as file:
        file.write(f"{y}\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(log_Q, "".join(seq), file) for seq in valid_seq(y, args.targeted)]
            concurrent.futures.wait(futures)
