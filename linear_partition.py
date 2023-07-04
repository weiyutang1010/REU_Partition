from collections import defaultdict
from heapq import nlargest
from score import paired, unpaired
import numpy as np

_allowed_pairs = {"AU", "UA", "CG", "GC", "GU", "UG"}

def partition_bu(x):
    """Bottom Up Approach"""
    n = len(x)
    Q = defaultdict(float)

    for i in range(n):
        Q[i,i-1] = 1.

    # O(n^3)
    for span in range(1, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1

            Q[i, j] = Q[i, j-1] * np.exp(-unpaired(x[j]))
            for k in range(i, j):
                if x[k] + x[j] in _allowed_pairs:
                    Q[i, j] += Q[i, k-1] * Q[k+1,j-1] * np.exp(-paired(x[k], x[j]))

    return Q

def partition_bu_log(x):
    """Bottom Up Approach"""
    n = len(x)
    Q = defaultdict(float)

    for i in range(n):
        Q[i,i-1] = 0.

    # O(n^3)
    for span in range(1, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1

            Q[i, j] = Q[i, j-1] + (-unpaired(x[j]))
            for k in range(i, j):
                if x[k] + x[j] in _allowed_pairs:
                    Q[i, j] = np.logaddexp(Q[i, j], Q[i, k-1] + Q[k+1,j-1] + (-paired(x[k], x[j])))

    return Q

def partition_lr(x):
    """Left to Right"""
    n = len(x)
    Q = [defaultdict(float) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 1.

    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] += Q[j-1][i] * np.exp(-unpaired(x[j-1]))

            # x is 0-indexed
            if i > 1 and x[i-2] + x[j-1] in _allowed_pairs:
                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i] * np.exp(-paired(x[i-2], x[j-1]))

    return Q

def partition_lr_log(x):
    """Left to Right"""
    n = len(x)
    Q = [defaultdict(lambda: float('1e-12')) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 0.

    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] = np.logaddexp(Q[j][i], Q[j-1][i] + (-unpaired(x[j-1])))

            # x is 0-indexed
            if i > 1 and x[i-2] + x[j-1] in _allowed_pairs:
                for k in Q[i-2]:
                    Q[j][k] = np.logaddexp(Q[j][k], Q[i-2][k] + Q[j-1][i] + (-paired(x[i-2], x[j-1])))

    return Q

def select_top_b(candidates, b):
    # heapq.nlargest is O(n log b)
    return set(nlargest(b, candidates, key=candidates.get))

def beam_prune(Q, j, b):
    candidates = defaultdict(float)

    for i in Q[j]:
        if 1 in Q[i-1]: # to prevent changing dict size during iteration
            candidates[i] = Q[i-1][1] * Q[j][i]

    candidates = select_top_b(candidates, b)

    for i in list(Q[j]):
        if i not in candidates:
            del Q[j][i]

def linear_partition(x, b):
    """Left to Right + Beam Pruning"""
    n = len(x)
    Q = [defaultdict(float) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 1.

    # O(nb^2)
    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] += Q[j-1][i] * np.exp(-unpaired(x[j-1]))

            # x is 0-indexed
            if i > 1 and x[i-2] + x[j-1] in _allowed_pairs:
                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i] * np.exp(-paired(x[i-2], x[j-1]))

        beam_prune(Q, j, b)

    return Q

def linear_partition_log(x, b):
    """Left to Right + Beam Pruning"""
    n = len(x)
    Q = [defaultdict(lambda: float('1e-12')) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 0.

    # O(nb^2)
    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] = np.logaddexp(Q[j][i], Q[j-1][i] + (-unpaired(x[j-1])))

            # x is 0-indexed
            if i > 1 and x[i-2] + x[j-1] in _allowed_pairs:
                for k in Q[i-2]:
                    Q[j][k] = np.logaddexp(Q[j][k], Q[i-2][k] + Q[j-1][i] + (-paired(x[i-2], x[j-1])))


        beam_prune(Q, j, b)

    return Q

def test_log():
    test_sequences = [
        "CG",
        "ACAGU",
        "CCGCG",
        "UUCAGGA",
        "ACGUACGU",
        "UUUGGCACUA",
        "AGGCAUCAAACCCUGCAUGGGAGCG",
    ]

    for seq in test_sequences:
        n = len(seq)
        print(f"Sequence: {seq}")
        print(f"Bottom-Up     Q(x) = {partition_bu_log(seq)[0, n-1]}")
        print(f"Left to Right Q(x) = {partition_lr_log(seq)[n][1]}")
        print(f"Beam Pruning  Q(x, b=1) = {linear_partition_log(seq, 1)[n][1]}")
        print(f"Beam Pruning  Q(x, b=3) = {linear_partition_log(seq, 3)[n][1]}")
        print(f"Beam Pruning  Q(x, b=5) = {linear_partition_log(seq, 5)[n][1]}")
        print(f"Beam Pruning  Q(x, b=10) = {linear_partition_log(seq, 10)[n][1]}")
        print(f"Beam Pruning  Q(x, b=50) = {linear_partition_log(seq, 50)[n][1]}")
        print("")

def test():
    test_sequences = [
        "CG",
        "ACAGU",
        "CCGCG",
        "UUCAGGA",
        "ACGUACGU",
        "UUUGGCACUA",
        "AGGCAUCAAACCCUGCAUGGGAGCG",
    ]

    for seq in test_sequences:
        n = len(seq)
        print(f"Sequence: {seq}")
        print(f"Bottom-Up     Q(x) = {partition_bu(seq)[0, n-1]}")
        print(f"Left to Right Q(x) = {partition_lr(seq)[n][1]}")
        print(f"Beam Pruning  Q(x, b=1) = {linear_partition(seq, 1)[n][1]}")
        print(f"Beam Pruning  Q(x, b=3) = {linear_partition(seq, 3)[n][1]}")
        print(f"Beam Pruning  Q(x, b=5) = {linear_partition(seq, 5)[n][1]}")
        print(f"Beam Pruning  Q(x, b=10) = {linear_partition(seq, 10)[n][1]}")
        print(f"Beam Pruning  Q(x, b=50) = {linear_partition(seq, 50)[n][1]}")
        print("")

if __name__ == "__main__":
    test()
    test_log()