from collections import defaultdict
from itertools import product
from linear_partition import partition_lr
import numpy as np

_allowed_pairs = {"AU", "UA", "CG", "GC", "GU", "UG"}
paired_sc = -1
unpaired_sc = .1

def linear_partition_lr(x):
    """Left to Right"""
    n = len(x)
    Q = [defaultdict(float) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 1.

    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] += Q[j-1][i] * np.exp(-unpaired_sc)

            # x is 0-indexed
            if i > 1 and x[i-2] + x[j-1] in _allowed_pairs:
                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i] * np.exp(-paired_sc)

    return Q[n][1]

def probability(seq, X):
    prob = 1
    for idx, c in enumerate(seq):
        prob *= X[idx][c]

    return prob

def verifier(X):
    # Let X be a distribution
    n = len(X)
    weighted_sum = 0

    for seq in product('ACGU', repeat=n):
        weighted_sum += probability(seq, X) * partition_lr(seq)

    return weighted_sum

def expected_partition(X):
    n = len(X)
    Q = [defaultdict(float) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 1.

    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] += Q[j-1][i] * np.exp(-unpaired_sc)

            # x is 0-indexed
            if i > 1 and x[i-2] + x[j-1] in _allowed_pairs:
                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i] * np.exp(-paired_sc)

    return Q[n][1]

def main():
    test_distribution_1 = [
        { 'A' : 0.2, 'C' : 0.3, 'G' : 0.1, 'U' : 0.4},
        { 'A' : 0.3, 'C' : 0.3, 'G' : 0.1, 'U' : 0.3},
        { 'A' : 0.5, 'C' : 0.2, 'G' : 0.2, 'U' : 0.1},
        { 'A' : 0.3, 'C' : 0.1, 'G' : 0.5, 'U' : 0.1},
    ]

    print("Expected Partition: \t", expected_partition(test_distribution_1))
    print("Answer: \t\t", verifier(test_distribution_1))

if __name__ == "__main__":
    main()