from collections import defaultdict
from itertools import product
from linear_partition import partition_lr
from score import paired, unpaired
import numpy as np

_allowed_pairs = {"AU", "UA", "CG", "GC", "GU", "UG"}
SMALL_NUM = -1000000

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
    """Left to Right"""
    n = len(X)
    Q = [defaultdict(float) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 1.

    for j in range(1, n+1):
        for i in Q[j-1]:
            unpaired_sc = 0.
            for c in X[j-1]: # c is {A, C, G, U}
                unpaired_sc += X[j-1][c] * np.exp(-unpaired(X, j))
            Q[j][i] += Q[j-1][i] * unpaired_sc

            if i > 1:
                for k in Q[i-2]:
                    paired_sc = 0.
                    for c1, c2 in _allowed_pairs:
                        paired_sc += X[i-2][c1] * X[j-1][c2] * np.exp(-paired(X, i-1, j))
                    Q[j][k] += Q[i-2][k] * Q[j-1][i] * paired_sc

    return Q[n][1]

def expected_partition_log(X):
    n = len(X)
    Q = [defaultdict(lambda: SMALL_NUM) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 0.

    for j in range(1, n+1):
        for i in Q[j-1]:
            unpaired_sc = SMALL_NUM
            for c in X[j-1]: # c is {A, C, G, U}
                unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j-1][c]) + (-unpaired(X, j)))
            Q[j][i] = np.logaddexp(Q[j][i], Q[j-1][i] + unpaired_sc)

            if i > 1:
                for k in Q[i-2]:
                    paired_sc = SMALL_NUM
                    for c1, c2 in _allowed_pairs:
                        paired_sc = np.logaddexp(paired_sc, np.log(X[i-2][c1]) + np.log(X[j-1][c2]) + (-paired(X, i-1, j)))
                    Q[j][k] = np.logaddexp(Q[j][k], Q[i-2][k] + Q[j-1][i] + paired_sc)

    return np.exp(Q[n][1])

def test():
    test_distribution_1 = [
        { 'A' : 0.2, 'C' : 0.3, 'G' : 0.1, 'U' : 0.4},
        { 'A' : 0.3, 'C' : 0.3, 'G' : 0.1, 'U' : 0.3},
    ]

    test_distribution_2 = [
        { 'A' : 0.2, 'C' : 0.3, 'G' : 0.1, 'U' : 0.4},
        { 'A' : 0.3, 'C' : 0.3, 'G' : 0.1, 'U' : 0.3},
        { 'A' : 0.5, 'C' : 0.2, 'G' : 0.2, 'U' : 0.1},
        { 'A' : 0.3, 'C' : 0.1, 'G' : 0.5, 'U' : 0.1},
    ]

    test_distribution_3 = [
        { 'A' : 0.5, 'C' : 0.2, 'G' : 0.2, 'U' : 0.1},
        { 'A' : 0.2, 'C' : 0.3, 'G' : 0.1, 'U' : 0.4},
        { 'A' : 0.3, 'C' : 0.3, 'G' : 0.1, 'U' : 0.3},
        { 'A' : 0.5, 'C' : 0.2, 'G' : 0.2, 'U' : 0.1},
        { 'A' : 0.3, 'C' : 0.3, 'G' : 0.1, 'U' : 0.3},
        { 'A' : 0.3, 'C' : 0.1, 'G' : 0.5, 'U' : 0.1},
    ]

    print("Test 1")
    print("Expected Partition: \t\t", expected_partition(test_distribution_1))
    print("Expected Partition (log): \t", expected_partition_log(test_distribution_1))
    print("Verifier: \t\t\t", verifier(test_distribution_1))
    print()

    print("Test 2")
    print("Expected Partition: \t\t", expected_partition(test_distribution_2))
    print("Expected Partition (log): \t", expected_partition_log(test_distribution_2))
    print("Verifier: \t\t\t", verifier(test_distribution_2))
    print()

    print("Test 3")
    print("Expected Partition: \t\t", expected_partition(test_distribution_3))
    print("Expected Partition (log): \t", expected_partition_log(test_distribution_3))
    print("Verifier: \t\t\t", verifier(test_distribution_3))

if __name__ == "__main__":
    test()