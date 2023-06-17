from collections import defaultdict
from itertools import product
from linear_partition import partition_lr
from score import paired, unpaired
import numpy as np
import math

_allowed_pairs = {"AU", "UA", "CG", "GC", "GU", "UG"}
SMALL_NUM = -1000000

sequences = {}

# Generate catesian product of input, https://docs.python.org/3/library/itertools.html#itertools.product
def generate_sequences(*args, n):
    pools = [tuple(pool) for pool in args] * n
    sequences[n] = [[]]
    for pool in pools:
        sequences[n] = [x+[y] for x in sequences[n] for y in pool]

def probability(seq, X):
    prob = 1
    for idx, c in enumerate(seq):
        prob *= X[idx][c]

    return prob

def verifier(X):
    # O(4^n * n^3)
    n = len(X)
    weighted_sum = 0
    
    if n not in sequences:
        generate_sequences('ACGU', n=n)
        
    for seq in sequences[n]:
        weighted_sum += probability(seq, X) * partition_lr(seq)

    return weighted_sum

def expected_partition(X):
    """Left to Right O(n^3)"""
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
    """O(n^3)"""
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

def generate_test_case(n):
    test_distribution = []

    for _ in range(n):
        # generate random number that sums to 1: https://stackoverflow.com/a/8068956
        rand = np.concatenate((np.random.sample(3), np.array([1., 0.])), axis=0)
        rand.sort()

        test_distribution.append({
            'A': rand[1] - rand[0],
            'C': rand[2] - rand[1],
            'G': rand[3] - rand[2],
            'U': rand[4] - rand[3]
        })

    return test_distribution


def test(n, t):
    np.random.seed(42)

    for _ in range(t):
        test_distribution = generate_test_case(n)
        exp = expected_partition(test_distribution)
        exp_log = expected_partition_log(test_distribution)
        ans = verifier(test_distribution)

        if not math.isclose(exp, ans):
            print(f"Wrong Value! expected partition       = {exp}, verifier = {ans}")

        if not math.isclose(exp_log, ans):
            print(f"Wrong Value! expected partition (log) = {exp}, verifier = {ans}")

    print(f"Completed test cases of n = {n}, t = {t}")

if __name__ == "__main__":
    for n in range(1, 21):
        test(n, 1)