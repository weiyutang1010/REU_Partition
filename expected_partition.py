from collections import defaultdict
from itertools import product
from inside_outside import inside_outside, inside_count, inside_prob, inside_prob_log, outside_forward_count, outside_forward_prob, outside_forward_prob_log
from score import paired, unpaired
from util import print_dicts, generate_sequences, generate_rand_distribution, probability, verify_dicts, RNA
import numpy as np
import math

RT = 1.
_allowed_pairs = {"AU", "UA", "CG", "GC", "GU", "UG"}
sequences = {}


def expected_verifier(X, inside_func, outside_func):
    # O(4^n * n^3)
    n = len(X)
    p = [defaultdict(float) for _ in range(n+1)]

    if n not in sequences:
        # Generate all 4^n sequences
        sequences[n] = generate_sequences('ACGU', n=n)
        
    Q_verify = [defaultdict(float) for _ in range(n+1)]
    Q_hat_verify = [defaultdict(float) for _ in range(n+1)]
    p_verify = [defaultdict(float) for _ in range(n+1)]
    
    for seq in sequences[n]:
        prob = probability(seq, X)
        Q, Q_hat, p = inside_outside(RNA(seq), inside_func, outside_func)
        
        for j, x in enumerate(Q):
            for i, value in x.items():
                Q_verify[j][i] += prob * value

        for j, x in enumerate(Q_hat):
            for i, value in x.items():
                Q_hat_verify[j][i] += prob * value

        for j, x in enumerate(p):
            for i, value in x.items():
                p_verify[j][i] += prob * value

    return Q_verify, Q_hat_verify, p_verify


def expected_inside_count(X):
    n = len(X)
    Q = [defaultdict(float) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 1.

    for j in range(1, n+1):
        for i in Q[j-1]:
            unpaired_sc = 0. # calculate weighted unpaired score
            for c in X[j]: # c is {A, C, G, U}
                unpaired_sc += X[j][c]
            Q[j][i] += Q[j-1][i] * unpaired_sc

            if i >= 2:
                paired_sc = 0.
                for c1, c2 in _allowed_pairs: # calculate weighted paired score
                    paired_sc += X[i-1][c1] * X[j][c2]

                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i] * paired_sc

    return Q

def expected_inside_partition(X):
    """Left to Right O(n^3)"""
    n = len(X)
    Q = [defaultdict(float) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 1.

    for j in range(1, n+1):
        for i in Q[j-1]:
            unpaired_sc = 0.
            # calculate weighted unpaired score
            for c in X[j]: # c is {A, C, G, U} 
                unpaired_sc += X[j][c] * np.exp(-unpaired(c)) 
            Q[j][i] += Q[j-1][i] * unpaired_sc

            if i > 1:
                paired_sc = 0.
                for c1, c2 in _allowed_pairs: # calculate weighted paired score
                    paired_sc += X[i-1][c1] * X[j][c2] * np.exp(-paired(c1, c2))

                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i] * paired_sc

    return Q

def expected_inside_partition_log(X):
    """X is 1-indexed"""
    n = len(X)
    Q = [defaultdict(lambda: float('-1e12')) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 0.

    Q[0][1] = 0.
    for j in range(1, n+1):
        for i in Q[j-1]:
            # unpaired_sc = float('-1e12') # calculate weighted unpaired score
            # for c in X[j]: # c is {A, C, G, U}
            #     unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j][c]) + (-unpaired(c)))
            unpaired_sc = -unpaired('A')
            Q[j][i] = np.logaddexp(Q[j][i], Q[j-1][i] + unpaired_sc)

            if i >= 2:
                paired_sc = float('-1e12') # calculate weighted paired score
                for c1, c2 in _allowed_pairs:
                    paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][c1]) + np.log(X[j][c2]) + (-paired(c1, c2)))
                
                for k in Q[i-2]:
                    Q[j][k] = np.logaddexp(Q[j][k], Q[i-2][k] + Q[j-1][i] + paired_sc)


    return Q


def expected_outside_count(Q, X):
    n = len(X)
    Q_hat = [defaultdict(float) for _ in range(n+1)]
    Q_hat[n][1] = 1.
    p = [defaultdict(float) for _ in range(n+1)]

    # X is 0 indexed, Q and Q_hat are 1 indexed
    for j in range(n, 0, -1):
        for i in Q[j-1]:
            unpaired_sc = 0. # calculate weighted unpaired score
            for c in X[j]: # c is {A, C, G, U}
                unpaired_sc += X[j][c]
            Q_hat[j-1][i] += Q_hat[j][i] * unpaired_sc

            if i >= 2:
                paired_sc = 0.
                for c1, c2 in _allowed_pairs: # calculate weighted paired score
                    paired_sc += X[i-1][c1] * X[j][c2]

                for k in Q[i-2]:
                    Q_hat[i-2][k] += Q_hat[j][k] * Q[j-1][i] * paired_sc
                    Q_hat[j-1][i] += Q_hat[j][k] * Q[i-2][k] * paired_sc
                    p[j][i-1] += (Q_hat[j][k] * Q[i-2][k] * Q[j-1][i] * paired_sc)

    return Q_hat, p

def expected_outside_partition(Q, X):
    n = len(X)
    Q_hat = [defaultdict(float) for _ in range(n+1)]
    Q_hat[n][1] = 1.
    p = [defaultdict(float) for _ in range(n+1)]

    for j in range(n, 0, -1):
        for i in Q[j-1]:
            unpaired_sc = np.exp(-unpaired('A') / RT)
            # for c in X[j]: # c is {A, C, G, U}
            #     unpaired_sc += X[j][c] * np.exp(-unpaired(c) / RT)  # calculate weighted unpaired score
            Q_hat[j-1][i] += Q_hat[j][i] * unpaired_sc

            if i > 1:
                paired_sc = 0.
                for c1, c2 in _allowed_pairs: # calculate weighted paired score
                    paired_sc += X[i-1][c1] * X[j][c2] * np.exp(-paired(c1, c2) / RT)

                for k in Q[i-2]:
                    Q_hat[i-2][k] += Q_hat[j][k] * Q[j-1][i] * paired_sc
                    Q_hat[j-1][i] += Q_hat[j][k] * Q[i-2][k] * paired_sc
                    p[j][i-1] += (Q_hat[j][k] * Q[i-2][k] * Q[j-1][i] * paired_sc) / Q[n][1]
    
    return Q_hat, p

def expected_outside_partition_log(Q, X):
    n = len(X)
    Q_hat = [defaultdict(lambda: float('-1e12')) for _ in range(n+1)]
    Q_hat[n][1] = 0.
    p = [defaultdict(lambda: float('-1e12')) for _ in range(n+1)]

    for j in range(n, 0, -1):
        for i in Q[j-1]:
            # unpaired_sc = float('-1e12')  # calculate weighted unpaired score
            # for c in X[j]: # c is {A, C, G, U}
            #     unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j][c]) + (-unpaired(c) / RT))
            unpaired_sc = -unpaired('A')
            Q_hat[j-1][i] = np.logaddexp(Q_hat[j-1][i], Q_hat[j][i] + unpaired_sc)

            if i >= 2:
                paired_sc = float('-1e12') # calculate weighted paired score
                for c1, c2 in _allowed_pairs:
                    paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][c1]) + np.log(X[j][c2]) + (-paired(c1, c2) / RT))

                for k in Q[i-2]:
                    Q_hat[i-2][k] = np.logaddexp(Q_hat[i-2][k], Q_hat[j][k] + Q[j-1][i] + paired_sc)
                    Q_hat[j-1][i] = np.logaddexp(Q_hat[j-1][i], Q_hat[j][k] + Q[i-2][k] + paired_sc)
                    p[j][i-1] = np.logaddexp(p[j][i-1], Q_hat[j][k] + Q[i-2][k] + Q[j-1][i] + paired_sc - Q[n][1])

    return Q_hat, p

def test_expected_count(n, t):
    for _ in range(t):
        X = RNA(generate_rand_distribution(n))
        
        Q = expected_inside_count(X)
        Q_hat, p = expected_outside_count(Q, X)

        Q_verify, Q_hat_verify, p_verify = expected_verifier(X, inside_count, outside_forward_count)
    
        if not verify_dicts(Q, Q_verify) or \
           not verify_dicts(Q_hat, Q_hat_verify) or \
           not verify_dicts(p, p_verify):
            print_dicts("Distribution", X)
            print_dicts("Q", Q)
            print_dicts("Q verify", Q_verify)
            print_dicts("Q_hat", Q_hat)
            print_dicts("Q_hat verify", Q_hat_verify)
            print_dicts("p", p)
            print_dicts("p verify", p_verify)
        
    print(f"Completed test cases of n = {n}, t = {t}")

def test_expected_partition(n, t):
    for _ in range(t):
        X = RNA(generate_rand_distribution(n))
        X = RNA([{'A': .25, 'C': .25, 'G': .25, 'U': .25} for i in range(n)])

        Q = expected_inside_partition(X)
        Q_hat, p = expected_outside_partition(Q, X)
        

        # TODO: Take exp of (Q_log and Q_hat_log) and compare to Q and Q_hat
        print_dicts("Distribution", X.seq)
        print_dicts("Q", Q)
        print_dicts("Q_hat", Q_hat)

    print(f"Completed test cases of n = {n}, t = {t}")

def test_expected_partition_log(n, t):
    for _ in range(t):
        X = RNA(generate_rand_distribution(n))
        X = RNA([{'A': .25, 'C': .25, 'G': .25, 'U': .25}, {'A': .25, 'C': .25, 'G': .25, 'U': .25}])

        Q = expected_inside_partition_log(X)
        Q_hat, p = expected_outside_partition_log(Q, X)
        
        Q_verify, Q_hat_verify, p_verify = expected_verifier(X, inside_prob_log, outside_forward_prob_log)

        if not verify_dicts(Q, Q_verify) or \
           not verify_dicts(Q_hat, Q_hat_verify):
        #    not verify_dicts(p, p_verify):
            print_dicts("Distribution", X.seq)
            print_dicts("Q", Q)
            # print_dicts("Q verify", Q_verify)
            print_dicts("Q_hat", Q_hat)
            # print_dicts("Q_hat verify", Q_hat_verify)

    print(f"Completed test cases of n = {n}, t = {t}")

def main():
    # X = RNA(generate_rand_distribution(n))
    X = RNA([{'A': .25, 'C': .25, 'G': .25, 'U': .25} for i in range(2)])

    Q = expected_inside_partition(X)
    Q_hat, p = expected_outside_partition(Q, X)
    
    Q_verify, Q_hat_verify, p_verify = expected_verifier(X, inside_prob, outside_forward_prob)

    print_dicts("Distribution", X.seq)
    print_dicts("Q", Q)
    print_dicts("Q verify", Q_verify)
    print_dicts("Q_hat", Q_hat)
    print_dicts("Q_hat verify", Q_hat_verify)


if __name__ == "__main__":
    # for n in range(2, 3):
    #     test_expected_partition(n, 1)
    #     test_expected_partition_log(n, 1)
        # test_expected_count(n, 10)
    # test_expected_partition_log(2, 1)
    # main()

    # X = RNA([{'A': .25, 'C': .25, 'G': .25, 'U': .25} for i in range(2)])

    X1 = RNA([{'A': .25, 'C': .25, 'G': .25, 'U': .25},
             {'A': .25, 'C': .25, 'G': .25, 'U': .25},
             {'A': .25, 'C': .25, 'G': .25, 'U': .25},
             {'A': .25, 'C': .25, 'G': .25, 'U': .25},])
    Q1 = expected_inside_partition(X1)

    X2 = RNA([{'A': .25, 'C': .25, 'G': .25, 'U': .25},
             {'A': .25, 'C': .25, 'G': .25, 'U': .25},
             {'A': .25001, 'C': .25, 'G': .25, 'U': .25},
             {'A': .25, 'C': .25, 'G': .25, 'U': .25},])
    Q2 = expected_inside_partition(X2)

    # Q = expected_inside_partition(X)
    # Q_hat, p = expected_outside_partition(Q, X)
    # print_dicts("Q", Q)
    # print_dicts("Q_hat", Q_hat)


    Q_hat1, p = expected_outside_partition(Q1, X1)
    print_dicts("Q1", Q1)

    Q_hat2, p = expected_outside_partition(Q2, X2)
    print_dicts("Q2", Q2)
    print_dicts("Q_hat1", Q_hat1)
    print_dicts("Q_hat2", Q_hat2)