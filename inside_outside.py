from collections import defaultdict
from score import paired, unpaired
from util import print_dicts, RNA, all_structs
import numpy as np
import math

_allowed_pairs = {"AU", "UA", "CG", "GC", "GU", "UG"}
RT = 1.

def inside_count(x):
    """Count the number of structure for each span i, j"""
    n = len(x)
    Q = [defaultdict(float) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 1.

    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] += Q[j-1][i]

            # x is 0-indexed
            if i > 1 and x[i-1] + x[j] in _allowed_pairs:
                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i]

    return Q

def inside_prob(x):
    """Left to Right"""
    n = len(x)
    Q = [defaultdict(float) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 1.

    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] += Q[j-1][i] * np.exp(-unpaired(x[j]))

            if i > 1 and x[i-1] + x[j] in _allowed_pairs:
                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i] * np.exp(-paired(x[i-1], x[j]))

    return Q

def inside_prob_log(x):
    """Left to Right"""
    n = len(x)
    Q = [defaultdict(lambda: float('-1e12')) for _ in range(n+1)]

    for j in range(1, n+1):
        Q[j-1][j] = 0.

    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] = np.logaddexp(Q[j][i], Q[j-1][i] + (-unpaired(x[j])))

            # x is 0-indexed
            if i >= 2 and x[i-1] + x[j] in _allowed_pairs:
                for k in Q[i-2]:
                    Q[j][k] = np.logaddexp(Q[j][k], Q[i-2][k] + Q[j-1][i] + (-paired(x[i-1], x[j])))

    return Q

def outside_forward_count(Q, x):
    n = len(x)
    Q_hat = [defaultdict(float) for _ in range(n+1)]
    Q_hat[n][1] = 1.
    p = [defaultdict(float) for _ in range(n+1)]

    for j in range(n, 0, -1):
        for i in Q[j-1]:
            Q_hat[j-1][i] += Q_hat[j][i]
            
            if i >= 2 and x[i-1] + x[j] in _allowed_pairs:
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
            Q_hat[j-1][i] += Q_hat[j][i] * np.exp(-unpaired(x[j]) / RT)
            
            if i >= 2 and x[i-1] + x[j] in _allowed_pairs:
                for k in Q[i-2]:
                    Q_hat[i-2][k] += Q_hat[j][k] * Q[j-1][i] * np.exp(-paired(x[i-1], x[j]) / RT)
                    Q_hat[j-1][i] += Q_hat[j][k] * Q[i-2][k] * np.exp(-paired(x[i-1], x[j]) / RT)
                    p[j][i-1] += (Q_hat[j][k] * Q[i-2][k] * Q[j-1][i] * np.exp(-paired(x[i-1], x[j]) / RT)) / Q[n][1]
    
    return Q_hat, p

def outside_forward_prob_log(Q, x):
    n = len(x)
    Q_hat = [defaultdict(lambda: float('-1e12')) for _ in range(n+1)]
    Q_hat[n][1] = 0.
    p = [defaultdict(lambda: float('-1e12')) for _ in range(n+1)]

    for j in range(n, 0, -1):
        for i in Q[j-1]:
            Q_hat[j-1][i] = np.logaddexp(Q_hat[j-1][i], Q_hat[j][i] + (-unpaired(x[j]) / RT))
            
            if i >= 2 and x[i-1] + x[j] in _allowed_pairs:
                for k in Q[i-2]:
                    Q_hat[i-2][k] = np.logaddexp(Q_hat[i-2][k], Q_hat[j][k] + Q[j-1][i] + (-paired(x[i-1], x[j]) / RT))
                    Q_hat[j-1][i] = np.logaddexp(Q_hat[j-1][i], Q_hat[j][k] + Q[i-2][k] + (-paired(x[i-1], x[j]) / RT))
                    p[j][i-1] = np.logaddexp(p[j][i-1], (Q_hat[j][k] + Q[i-2][k] + Q[j-1][i] + (-paired(x[i-1], x[j]) / RT)) - Q[n][1])
    
    return Q_hat, p

def inside_outside(x, inside_func, outside_func):
    Q = inside_func(x)
    Q_hat, p = outside_func(Q, x)
    return Q, Q_hat, p

def test_count(n ,t):
    for _ in range(t):
        seq = RNA("".join([np.random.choice(['A', 'C', 'G', 'U']) for _ in range(n)]))
        Q ,Q_hat, p = inside_outside(seq, inside_count, outside_forward_count)

        # Verifying
        structs = all_structs(seq.seq)
        num_pairs = [defaultdict(float) for _ in range(n+1)]
        for num_p, struct in structs:
            stack = []
            for j in range(n):
                if struct[j] == '(':
                    stack.append(j)
                elif struct[j] == ')':
                    i = stack.pop()
                    num_pairs[j+1][i+1] += 1.0

        if p != num_pairs:
            print("Wrong Value!")

            print(seq)
            print()

            print("Q: ")
            for i, x in enumerate(Q):
                print(i, dict(x))
            print()

            print("Q_hat: ")
            for i, x in enumerate(Q_hat):
                print(i, dict(x))
            print()

            print("p: ")
            for i, x in enumerate(p):
                print(i, dict(x))
            print()

            print("ans: ")
            for i, x in enumerate(num_pairs):
                print(i, dict(x))
            print()

    print(f"Completed test cases of n = {n}, t = {t}")

def test_prob(n, t):
    for _ in range(t):
        seq = RNA("".join([np.random.choice(['A', 'C', 'G', 'U']) for _ in range(n)]))
        Q ,Q_hat, p = inside_outside(seq, inside_prob, outside_forward_prob)
        Q_log ,Q_hat_log, p_log = inside_outside(seq, inside_prob_log, outside_forward_prob_log)

        # Verifying
        structs = all_structs(seq.seq)
        pairs_prob = [defaultdict(float) for _ in range(n+1)]

        for _, struct in structs:
            stack = []
            pairs = []
            delta_G = .0

            for j in range(n):
                if struct[j] == '(':
                    stack.append(j)
                elif struct[j] == ')':
                    i = stack.pop()
                    pairs.append((i, j))

                    delta_G += paired(seq[i+1], seq[j+1])
                else:
                    delta_G += unpaired(seq[j+1])

            for i, j in pairs:
                pairs_prob[j+1][i+1] += np.exp(-delta_G / RT) / Q[n][1]
        
        close = True
        for j, x in enumerate(pairs_prob):
            for i in x:
                if not math.isclose(pairs_prob[j][i], p[j][i]):
                    close = False
                    break

                if not math.isclose(pairs_prob[j][i], np.exp(p_log[j][i])):
                    close = False
                    break

        if not close:
            print("Wrong Value!")

            print("seq: ", seq)

            print("p: ")
            for i, x in enumerate(p):
                print(i, dict(x))

            print("p_log: ")
            for i, x in enumerate(p_log):
                print(i, dict(x))

            print("ans: ")
            for i, x in enumerate(pairs_prob):
                print(i, dict(x))

    print(f"Completed test cases of n = {n}, t = {t}")


if __name__ == '__main__':
    t = 100
    for n in range(1, 11):
        test_count(n, t)
        test_prob(n, t)

