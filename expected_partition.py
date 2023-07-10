from collections import defaultdict
from itertools import product
from inside_partition import partition_lr
from outside_partition import inside_outside, outside_forward_prob
from score import paired, unpaired
import numpy as np
import math

RT = 1.
_allowed_pairs = {"AU", "UA", "CG", "GC", "GU", "UG"}
sequences = {}

def print_dicts(text, array):
    # Print a list of dicts
    print(text)
    for i, x in enumerate(array):
        print(i, dict(x))
    print()

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

def expected_inside_verifier(X):
    # O(4^n * n^3)
    n = len(X)
    weighted_sum = 0
    
    if n not in sequences:
        generate_sequences('ACGU', n=n)
        
    for seq in sequences[n]:
        weighted_sum += probability(seq, X) * partition_lr(seq)[n][1]

    return weighted_sum

def expected_outside_verifier(Q, X):
    # O(4^n * n^3)
    n = len(X)
    p = [defaultdict(float) for _ in range(n+1)]

    if n not in sequences:
        generate_sequences('ACGU', n=n)
        
    p_verify = [defaultdict(float) for _ in range(n+1)]
    for seq in sequences[n]:
        prob = probability(seq, X)
        Q, Q_hat, p = inside_outside(seq, partition_lr, outside_forward_prob)

        for j in range(n+1):
            for i in range(1, j):
                p_verify[j][i] += p[j][i]

    return p_verify


def expected_inside_partition(X):
    """Left to Right O(n^3)"""
    n = len(X)
    Q = [defaultdict(float) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 1.

    for j in range(1, n+1):
        for i in Q[j-1]:
            unpaired_sc = 0. # calculate weighted unpaired score
            for c in X[j-1]: # c is {A, C, G, U}
                unpaired_sc += X[j-1][c] * np.exp(-unpaired(c))
            Q[j][i] += Q[j-1][i] * unpaired_sc

            if i > 1:
                paired_sc = 0.
                for c1, c2 in _allowed_pairs: # calculate weighted paired score
                    paired_sc += X[i-2][c1] * X[j-1][c2] * np.exp(-paired(c1, c2))

                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i] * paired_sc

    return Q

def expected_inside_partition_log(X):
    """O(n^3)"""
    n = len(X)
    Q = [defaultdict(lambda: float('-1e12')) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 0.

    for j in range(1, n+1):
        for i in Q[j-1]:
            unpaired_sc = float('-1e12') # calculate weighted unpaired score
            for c in X[j-1]: # c is {A, C, G, U}
                unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j-1][c]) + (-unpaired(c)))
            Q[j][i] = np.logaddexp(Q[j][i], Q[j-1][i] + unpaired_sc)

            if i > 1:
                paired_sc = float('-1e12') # calculate weighted paired score
                for c1, c2 in _allowed_pairs:
                    paired_sc = np.logaddexp(paired_sc, np.log(X[i-2][c1]) + np.log(X[j-1][c2]) + (-paired(c1, c2)))
                
                for k in Q[i-2]:
                    Q[j][k] = np.logaddexp(Q[j][k], Q[i-2][k] + Q[j-1][i] + paired_sc)

    return Q

def expected_outside_partition(Q, X):
    n = len(X)
    Q_hat = [defaultdict(float) for _ in range(n+1)]
    Q_hat[n][1] = 1.
    p = [defaultdict(float) for _ in range(n+1)]

    # X is 0 indexed, Q and Q_hat are 1 indexed
    for j in range(n, 0, -1):
        for i in Q[j-1]:
            unpaired_sc = 0. # calculate weighted unpaired score
            for c in X[j-1]: # c is {A, C, G, U}
                unpaired_sc += X[j-1][c] * np.exp(-unpaired(c) / RT)
            Q_hat[j-1][i] += Q_hat[j][i] * unpaired_sc

            if i >= 2:
                paired_sc = 0.
                for c1, c2 in _allowed_pairs: # calculate weighted paired score
                    paired_sc += X[i-2][c1] * X[j-1][c2] * np.exp(-paired(c1, c2) / RT)

                for k in Q[i-2]:
                    Q_hat[i-2][k] += Q_hat[j][k] * Q[j-1][i] * paired_sc
                    Q_hat[j-1][i] += Q_hat[j][k] * Q[i-2][k] * paired_sc
                    p[j][i-1] += (Q_hat[j][k] * Q[i-2][k] * Q[j-1][i] * paired_sc) / Q[n][1]

    return Q_hat, p

def generate_test_case(n):
    test_distribution = []

    for _ in range(n):
        # generate random numbers that sum to 1: https://stackoverflow.com/a/8068956
        rand = np.concatenate((np.random.sample(3), np.array([1., 0.])), axis=0)
        rand.sort()

        test_distribution.append({
            'A': rand[1] - rand[0],
            'C': rand[2] - rand[1],
            'G': rand[3] - rand[2],
            'U': rand[4] - rand[3]
        })

    return test_distribution

def test_inside(n, t):
    for _ in range(t):
        test_distribution = generate_test_case(n)
        exp = expected_inside_partition(test_distribution)
        exp_log = expected_inside_partition_log(test_distribution)
        ans = expected_inside_verifier(test_distribution)

        if not math.isclose(exp[n][1], ans):
            print(f"Wrong Value! expected partition       = {exp}, verifier = {ans}")

        if not math.isclose(np.exp(exp_log[n][1]), ans):
            print(f"Wrong Value! expected partition (log) = {exp}, verifier = {ans}")

    print(f"Completed test cases of n = {n}, t = {t}")

def test_outside(n, t):
    for _ in range(t):
        # test_distribution = generate_test_case(n)
        test_distribution = [{'A': .25, 'C': .25, 'G': .25, 'U': .25} for _ in range(n)]
        Q = expected_inside_partition(test_distribution)
        Q_hat, p = expected_outside_partition(Q, test_distribution)
        p_verify = expected_outside_verifier(Q, test_distribution)

        print_dicts("Test Distribution", test_distribution)
        print_dicts("Expected Partition (Inside)", Q)
        print_dicts("Expected Partition (Outside)", Q_hat)
        print_dicts("Expected Pairing Probability", p)
        print_dicts("Verify", p_verify)

if __name__ == "__main__":
    np.random.seed(42)
    # for n in range(10):
    #     test_inside(n, 100)
    test_outside(2, 1)