from collections import defaultdict
from linear_partition import partition_lr
from score import paired, unpaired
from rna import total, kbest
import numpy as np

_allowed_pairs = {"AU", "UA", "CG", "GC", "GU", "UG"}

def inside_count(x):
    """Left to Right"""
    n = len(x)
    Q = [defaultdict(float) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 1.

    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] += Q[j-1][i]

            # x is 0-indexed
            if i > 1 and x[i-2] + x[j-1] in _allowed_pairs:
                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i]

    return Q

def outside_forward_count(Q, x):
    n = len(x)
    Q_hat = [defaultdict(float) for _ in range(n+1)]
    Q_hat[n][1] = 1.
    p = [defaultdict(float) for _ in range(n+1)]

    for j in range(n, 0, -1):
        for i in Q[j-1]:
            Q_hat[j-1][i] += Q_hat[j][i]
            
            if i >= 2 and x[i-2] + x[j-1] in _allowed_pairs:
                for k in Q[i-2]:
                    Q_hat[i-2][k] += Q_hat[j][k] * Q[j-1][i]
                    Q_hat[j-1][i] += Q_hat[j][k] * Q[i-2][k]
                    p[j][i-1] += Q_hat[j][k] * Q[i-2][k] * Q[j-1][i]
    
    return Q_hat, p



def outside_forward_prob(Q, x):
    n = len(x)
    Q_hat = [defaultdict(float) for _ in range(n+1)]
    Q_hat[n][1] = 1.
    p = [defaultdict(float) for _ in range(n+1)]

    for j in range(n, 0, -1):
        for i in Q[j-1]:
            Q_hat[j-1][i] += Q_hat[j][i]
            
            if i >= 2 and x[i-2] + x[j-1] in _allowed_pairs:
                for k in Q[i-2]:
                    Q_hat[i-2][k] += Q_hat[j][k] * Q[j-1][i]
                    Q_hat[j-1][i] += Q_hat[j][k] * Q[i-2][k]
                    p[j][i-1] += Q_hat[j][k] * Q[i-2][k] * Q[j-1][i]
    
    return Q_hat, p

def inside_outside(x, inside_func, outside_func):
    Q = inside_func(x)
    Q_hat, p = outside_func(Q, x)
    return Q, Q_hat, p

def test_count(n ,t):
    for _ in range(t):
        seq = "".join([np.random.choice(['A', 'C', 'G', 'U']) for _ in range(n)])
        Q ,Q_hat, p = inside_outside(seq, inside_count, outside_forward_count)
        all_structs = kbest(seq, 100000)

        num_pairs = [defaultdict(float) for _ in range(n+1)]
        for num_p, struct in all_structs:
            stack = []
            for j in range(n):
                if struct[j] == '(':
                    stack.append(j)
                elif struct[j] == ')':
                    i = stack.pop()
                    num_pairs[j+1][i+1] += 1.0

        if p != num_pairs:
            print("Wrong Value!")

            print("p: ")
            for i, x in enumerate(p):
                print(i, dict(x))

            print("ans: ")
            for i, x in enumerate(num_pairs):
                print(i, dict(x))

    print(f"Completed test cases of n = {n}, t = {t}")

def main():
    test_sequences = [
        "ACAGU",
        "CCAAAGG"
    ]
    
    for seq in test_sequences:
        print(seq)
        Q ,Q_hat, p = inside_outside(seq, inside_count, outside_forward_count)
        all_structs = kbest(seq, 1000000)

        # Inside partition
        print("Inside Partition: ")
        for i, x in enumerate(Q):
            print(i, dict(x))
        print()

        # Outside partition
        print("Outside Partition: ")
        for i, x in enumerate(Q_hat):
            print(i, dict(x))
        print()

        # Pairing
        print("Pairs: ")
        for i, x in enumerate(p):
            print(i, dict(x))
        print()

        print(all_structs)
        print()

if __name__ == '__main__':
    t = 100
    for n in range(1, 10):
        test_count(n, t)
    main()