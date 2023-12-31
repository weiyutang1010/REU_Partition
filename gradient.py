from collections import defaultdict
from itertools import product
from inside_outside import inside_outside, inside_count, inside_prob, inside_prob_log, outside_forward_count, outside_forward_prob, outside_forward_prob_log
from score import paired, unpaired
from util import dicts_to_lists, generate_sequences, generate_rand_structure, generate_rand_distribution, probability, verify_dicts, RNA
import dynet as dy
import numpy as np
import math

RT = 1.
_allowed_pairs = {"AU", "UA", "CG", "GC", "GU", "UG"}
_invalid_pairs = {"AA", "AC", "AG", "CA", "CC", "CU", "GA", "GG", "UC", "UU"}
NUC_TO_NUM = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
NEG_INF = '-1e18'

def dynet_inside_partition(X, sharpturn=0):
    """X is a n x 4 matrix"""
    n = X.dim()[0][0]
    Q = defaultdict(lambda: defaultdict(lambda: dy.scalarInput(0.)))

    for j in range(n):
        Q[j-1][j]= dy.scalarInput(1.)

    for j in range(n):
        for i in Q[j-1]:
            unpaired_sc = dy.scalarInput(0.)
            for c in NUC_TO_NUM: # c is {A, C, G, U}
                unpaired_sc += X[j][NUC_TO_NUM[c]] * np.exp(-unpaired(c) / RT)
            Q[j][i] += Q[j-1][i] * unpaired_sc

            if i > 0 and j-(i-1) > sharpturn:
                paired_sc = dy.scalarInput(0.)
                for c1, c2 in _allowed_pairs:
                    paired_sc += X[i-1][NUC_TO_NUM[c1]] * X[j][NUC_TO_NUM[c2]] * dy.exp(dy.scalarInput(-paired(c1, c2) / RT))

                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i] * paired_sc

    return Q


def expected_inside_partition(X, sharpturn=0):
    """Left to Right O(n^3), 0-indexed"""
    n = len(X)
    Q_hat = defaultdict(lambda: defaultdict(float))

    for j in range(n):
        Q_hat[j-1][j] = 1.

    for j in range(n):
        for i in Q_hat[j-1]:
            unpaired_sc = 0.
            for c in X[j]: # c is {A, C, G, U} 
                unpaired_sc += X[j][c] * np.exp(-unpaired(c)/RT) 
            Q_hat[j][i] += Q_hat[j-1][i] * unpaired_sc

            if i > 0 and j-(i-1) > sharpturn:
                paired_sc = 0.
                for c1, c2 in _allowed_pairs: # calculate weighted paired score
                    paired_sc += X[i-1][c1] * X[j][c2] * np.exp(-paired(c1, c2))

                for k in Q_hat[i-2]:
                    Q_hat[j][k] += Q_hat[i-2][k] * Q_hat[j-1][i] * paired_sc

    return Q_hat

def expected_outside_partition(Q_hat, X, sharpturn=0):
    """Top-Down, 0-indexed"""
    n = len(X)
    O_hat = defaultdict(lambda: defaultdict(float))
    gradient = [{'A': 0., 'C': 0., 'G': 0., 'U': 0.} for _ in range(n)]

    O_hat[n-1][0] = 1.

    for j in range(n-1, -1, -1):
        for i in Q_hat[j-1]:
            unpaired_sc = 0.
            for nucj in X[j]: # nucj is {A, C, G, U}
                unpaired_sc += X[j][nucj] * np.exp(-unpaired(nucj) / RT)  # calculate weighted unpaired score
                gradient[j][nucj] += O_hat[j][i] * Q_hat[j-1][i] * np.exp(-unpaired(nucj) / RT)
            O_hat[j-1][i] += O_hat[j][i] * unpaired_sc
                
            if i > 0 and j-(i-1) > sharpturn:
                paired_sc = 0.
                for c1, c2 in _allowed_pairs: # calculate weighted paired score
                    paired_sc += X[i-1][c1] * X[j][c2] * np.exp(-paired(c1, c2) / RT)

                for k in Q_hat[i-2]:
                    O_hat[i-2][k] += O_hat[j][k] * Q_hat[j-1][i] * paired_sc
                    O_hat[j-1][i] += O_hat[j][k] * Q_hat[i-2][k] * paired_sc

                    grad = O_hat[j][k] * Q_hat[j-1][i] * Q_hat[i-2][k]
                    for nuci_1, nucj in _allowed_pairs:
                        gradient[i-1][nuci_1] += X[j][nucj] * np.exp(-paired(nuci_1, nucj)) * grad
                        gradient[j][nucj] += X[i-1][nuci_1] * np.exp(-paired(nuci_1, nucj)) * grad

    return O_hat, gradient

def dynet_inside_partition_log(X, sharpturn=0):
    """0-indexed"""
    n = X.dim()[0][0]
    Q_hat = defaultdict(lambda: defaultdict(lambda: dy.scalarInput(float(NEG_INF))))

    for j in range(n):
        Q_hat[j-1][j]= dy.scalarInput(0.)

    for j in range(n):
        for i in Q_hat[j-1]:
            unpaired_sc = dy.scalarInput(float(NEG_INF))
            for c in NUC_TO_NUM:
                unpaired_sc = dy.logsumexp([unpaired_sc, dy.log(X[j][NUC_TO_NUM[c]]) + (-unpaired(c) / RT)])
            Q_hat[j][i] = dy.logsumexp([Q_hat[j][i], Q_hat[j-1][i] + unpaired_sc])

            if i > 0 and j-(i-1) > sharpturn:
                paired_sc = dy.scalarInput(float(NEG_INF))
                for c1, c2 in _allowed_pairs:
                    nuci_1 = NUC_TO_NUM[c1]
                    nucj = NUC_TO_NUM[c2]
                    paired_sc = dy.logsumexp([paired_sc, dy.log(X[i-1][nuci_1]) + dy.log(X[j][nucj]) + dy.scalarInput(-paired(c1, c2) / RT)])

                for k in Q_hat[i-2]:
                    Q_hat[j][k] = dy.logsumexp([Q_hat[j][k], Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc])

    return Q_hat    

def expected_inside_partition_log(X, sharpturn=0):
    n = len(X)
    Q_hat = defaultdict(lambda: defaultdict(lambda: float(NEG_INF)))

    for j in range(n):
        Q_hat[j-1][j] = 0.

    for j in range(n):
        for i in Q_hat[j-1]:
            unpaired_sc = float(NEG_INF)
            for c in X[j]:
                unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j][c]) + (-unpaired(c) / RT))
            Q_hat[j][i] = np.logaddexp(Q_hat[j][i], Q_hat[j-1][i] + unpaired_sc)

            if i > 0 and j-(i-1) > sharpturn:
                paired_sc = float(NEG_INF) # calculate weighted paired score
                for c1, c2 in _allowed_pairs:
                    paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][c1]) + np.log(X[j][c2]) + (-paired(c1, c2)))
                
                for k in Q_hat[i-2]:
                    Q_hat[j][k] = np.logaddexp(Q_hat[j][k], Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc)

    return Q_hat


def expected_outside_partition_log(Q_hat, X, sharpturn=0):
    """Backpropagation of Expected Inside Partition. Q_hat, O_hat, and gradient are calculated in log space.
       The dynet result is given by exp(gradient)."""
    n = len(X)
    O_hat = defaultdict(lambda: defaultdict(lambda: float(NEG_INF)))
    gradient = [{'A': float(NEG_INF), 'C': float(NEG_INF), 'G': float(NEG_INF), 'U': float(NEG_INF)} for _ in range(n)]

    O_hat[n-1][0] = 0.
    for j in range(n-1, -1, -1):
        for i in Q_hat[j-1]:
            unpaired_sc = float(NEG_INF) # calculate weighted unpaired score
            for c in X[j]: 
                unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j][c]) + (-unpaired(c) / RT))
            O_hat[j-1][i] = np.logaddexp(O_hat[j-1][i], O_hat[j][i] + Q_hat[j-1][i] + unpaired_sc - Q_hat[j][i])

            for c in X[j]:
                # gradient of Q_hat[1][n] with respect to unpaired_sc
                grad = O_hat[j][i] + Q_hat[j-1][i] + unpaired_sc - Q_hat[j][i]
                # gradient of unpaired_sc with respect to nucj
                softmax = np.log(X[j][c]) + (-unpaired(c) / RT) - unpaired_sc
                gradient[j][c] =  np.logaddexp(gradient[j][c], grad - np.log(X[j][c]) + softmax)

            if i > 0 and j-(i-1) > sharpturn:
                paired_sc = float(NEG_INF) # calculate weighted paired score
                for c1, c2 in _allowed_pairs:
                    paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][c1]) + np.log(X[j][c2]) + (-paired(c1, c2) / RT))

                for k in Q_hat[i-2]:
                    O_hat[i-2][k] = np.logaddexp(O_hat[i-2][k], O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k])
                    O_hat[j-1][i] = np.logaddexp(O_hat[j-1][i], O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k])

                    # gradient of Q_hat[1][n] with respect to paired_sc
                    grad = O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k]

                    # gradient of Q_hat[1][n] with respect to nuci_1, nucj
                    for nuci_1, nucj in _allowed_pairs:
                        softmax = np.log(X[i-1][nuci_1]) + np.log(X[j][nucj]) + (-paired(nuci_1, nucj)) - paired_sc
                        gradient[i-1][nuci_1] = np.logaddexp(gradient[i-1][nuci_1], grad - np.log(X[i-1][nuci_1]) + softmax)
                        gradient[j][nucj] = np.logaddexp(gradient[j][nucj], grad - np.log(X[j][nucj]) + softmax)

    return O_hat, gradient

def dynet_free_energy(struct, X):
    n = len(struct)
    delta_G = dy.scalarInput(0.)
    penalty = dy.scalarInput(10.)

    stack = []
    for j in range(n):
        if struct[j] == '(':
            stack.append(j)
        elif struct[j] == ')':
            i = stack.pop()

            score_ij = dy.scalarInput(0.)
            
            for c1, c2 in _allowed_pairs:
                nuci, nucj = NUC_TO_NUM[c1], NUC_TO_NUM[c2]
                score_ij += X[i][nuci] * X[j][nucj] * paired(c1, c2)

            for c1, c2 in _invalid_pairs:
                nuci, nucj = NUC_TO_NUM[c1], NUC_TO_NUM[c2]
                score_ij += X[i][nuci] * X[j][nucj] * penalty

            delta_G += score_ij
        else:
            delta_G += unpaired()

    return delta_G

def expected_free_energy(struct, X):
    n = len(struct)
    free_energy = 0.
    penalty = 10.

    gradient = [{'A': 0., 'C': 0., 'G': 0., 'U': 0.} for _ in range(n)]

    stack = []
    for j in range(n):
        if struct[j] == '(':
            stack.append(j)
        elif struct[j] == ')':
            i = stack.pop()

            score_ij = 0.
            
            for nuci, nucj in _allowed_pairs:
                score_ij += X[i][nuci] * X[j][nucj] * paired(nuci, nucj)
                gradient[i][nuci] += X[j][nucj] * paired(nuci, nucj)
                gradient[j][nucj] += X[i][nuci] * paired(nuci, nucj)

            for nuci, nucj in _invalid_pairs:
                score_ij += X[i][nuci] * X[j][nucj] * penalty
                gradient[i][nuci] += X[j][nucj] * penalty
                gradient[j][nucj] += X[i][nuci] * penalty

            free_energy += score_ij
        else:
            free_energy += unpaired()

    return free_energy, gradient


def verify_gradient(grad1, grad2, n, rel_tol=float('1e-06'), abs_tol=0):
    for j in range(n):
        for nuc in range(4):
            if not math.isclose(grad1[j][nuc], grad2[j][nuc], rel_tol=rel_tol, abs_tol=abs_tol):
                print(grad1[j][nuc], grad2[j][nuc])
                return False
    return True

def print_dicts(label, dicts):
    print(label)
    for j in sorted(dicts):
        for i in sorted(dicts[j]):
            print(f"({i}, {j})\t: {dicts[j][i]: .5f}\t", end="")
        print()
    print()


def test_dynet(n, t):
    m = dy.ParameterCollection()
    ts = m.add_parameters((n, 4))

    for _ in range(t):
        X = generate_rand_distribution(n)
        X1 = dicts_to_lists(X)
        ts.set_value(X1)

        dy.renew_cg()
        x = dy.scalarInput(0.)
        ts_input = ts + x
        inside = dynet_inside_partition(ts_input)

        inside[n-1][0].backward()
        dy_gradient = ts.gradient()

        Q_hat = expected_inside_partition(X)
        O_hat, gradient = expected_outside_partition(Q_hat, X)
        gradient = dicts_to_lists(gradient)

        if not verify_gradient(dy_gradient, gradient, n, abs_tol=float('1e-4')):
            print("Wrong Values")
            print(dy_gradient, end="\n\n")
            print(gradient, end="\n\n")
    
    print(f"Test Done: n={n}, t={t}")


def test_dynet_log(n, t):
    m = dy.ParameterCollection()
    ts = m.add_parameters((n, 4))

    for _ in range(t):
        X = generate_rand_distribution(n)

        X1 = dicts_to_lists(X)
        ts.set_value(X1)

        dy.renew_cg()
        x = dy.scalarInput(0.)
        ts_input = ts + x
        inside = dynet_inside_partition_log(ts_input)

        inside[n-1][0].backward()
        dy_gradient = ts.gradient()

        Q_hat = expected_inside_partition_log(X)
        O_hat, gradient = expected_outside_partition_log(Q_hat, X)
        gradient = np.exp(dicts_to_lists(gradient))

        if not verify_gradient(dy_gradient, gradient, n, abs_tol=float('1e-4')):
            print("Wrong Values")
            print(dy_gradient, end="\n\n")
            print(gradient, end="\n\n")
    
    print(f"Test Done: n={n}, t={t}")

def test_free_energy(n, t):
    m = dy.ParameterCollection()
    ts = m.add_parameters((n, 4))

    for _ in range(t):
        X = generate_rand_distribution(n)
        struct = generate_rand_structure(n)

        X1 = dicts_to_lists(X)
        ts.set_value(X1)

        dy.renew_cg()
        x = dy.scalarInput(0.)
        ts_input = ts + x
        delta_G = dynet_free_energy(struct, ts_input)

        delta_G.backward()
        dy_gradient = ts.gradient()

        free_energy, gradient = expected_free_energy(struct, X)
        gradient = dicts_to_lists(gradient)
        
        if not verify_gradient(dy_gradient, gradient, n, abs_tol=float('1e-5')):
            print("Wrong Values")
            print(dy_gradient, end="\n\n")
            print(gradient, end="\n\n")

    print(f"Test Done: n={n}, t={t}")

def main():
    X = [{'A': .25, 'C': .25, 'G': .25, 'U': .25},
         {'A': .25, 'C': .25, 'G': .25, 'U': .25},
         {'A': .25, 'C': .25, 'G': .25, 'U': .25},
         {'A': .25, 'C': .25, 'G': .25, 'U': .25},]

    n = len(X)

    Q_hat = expected_inside_partition(X)
    O_hat, gradient = expected_outside_partition(Q_hat, X)

    print_dicts('Expected Inside Partition', Q_hat)
    print_dicts('Expected Outside Partition', O_hat)
    gradient = dicts_to_lists(gradient)
    print("Gradient")
    print(gradient, end="\n\n")

    # Q_hat_log = expected_inside_partition_log(X)
    # O_hat_log, gradient_log = expected_outside_partition_log(Q_hat_log, X)

    # print_dicts('Expected Inside Partition (log)', Q_hat_log)
    # print_dicts('Expected Outside Partition (log)', O_hat_log)
    # gradient_log = np.exp(dicts_to_lists(gradient_log))
    # print(gradient_log, end="\n\n")
    

if __name__ == '__main__':
    # main()
    for n in range(1, 20):
        # test_dynet(n, 10)
        test_dynet_log(n, 10)
        # test_free_energy(n, 10)


  




def expected_outside_partition_log(Q_hat, X, sharpturn=0):
    """Q_hat is in log-space, O_hat and gradient are not. Backpropagation of expected inside log."""
    n = len(X)
    O_hat = defaultdict(lambda: defaultdict(lambda: float(0.)))
    gradient = [{'A': 0., 'C': 0., 'G': 0., 'U': 0.} for _ in range(n)]

    O_hat[n-1][0] = 1.
    for j in range(n-1, -1, -1):
        for i in Q_hat[j-1]:
            unpaired_sc = float(NEG_INF) # calculate weighted unpaired score
            for c in X[j]: # this part could be wrong (not verified)
                unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j][c]) + (-unpaired(c) / RT))
            O_hat[j-1][i] += O_hat[j][i] * (np.exp(Q_hat[j-1][i] + unpaired_sc) / np.exp(Q_hat[j][i]))

            for c in X[j]:
                # gradient of Q_hat[1][n] with respect to unpaired_sc
                grad = O_hat[j][i] * (np.exp(Q_hat[j-1][i] + unpaired_sc) / np.exp(Q_hat[j][i]))
                # gradient of unpaired_sc with respect to nucj
                softmax = np.exp(np.log(X[j][c]) + (-unpaired(c) / RT)) / np.exp(unpaired_sc)
                gradient[j][c] +=  grad * (1 / X[i-1][nuci_1]) * softmax

            if i > 0 and j-(i-1) > sharpturn:
                paired_sc = float(NEG_INF) # calculate weighted paired score
                for c1, c2 in _allowed_pairs:
                    paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][c1]) + np.log(X[j][c2]) + (-paired(c1, c2) / RT))

                for k in Q_hat[i-2]:
                    O_hat[i-2][k] += O_hat[j][k] * (np.exp(Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc) / np.exp(Q_hat[j][k]))
                    O_hat[j-1][i] += O_hat[j][k] * (np.exp(Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc) / np.exp(Q_hat[j][k]))

                    # gradient of Q_hat[1][n] with respect to paired_sc
                    grad = O_hat[j][k] * (np.exp(Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc) / np.exp(Q_hat[j][k])) 

                    # gradient of paired_sc with respect to nuci_1, nucj
                    for nuci_1, nucj in _allowed_pairs:
                        softmax = np.exp(np.log(X[i-1][nuci_1]) + np.log(X[j][nucj]) + (-paired(nuci_1, nucj))) / np.exp(paired_sc)
                        gradient[i-1][nuci_1] += grad * (1 / X[i-1][nuci_1]) * softmax
                        gradient[j][nucj] += grad * (1 / X[j][nucj]) * softmax

    return O_hat, gradient


# def expected_outside_partition_log(Q, X):
#     """Calculate log Q_hat and log gradient"""
#     n = len(X)
#     Q_hat = [defaultdict(lambda: float(NEG_INF)) for _ in range(n+1)]
#     grad = [defaultdict(lambda: float(NEG_INF)) for _ in range(n+1)]

#     Q_hat[n][1] = 1.
#     for j in range(n, 0, -1):
#         for i in Q[j-1]:
#             unpaired_sc = -unpaired()
#             Q_hat[j-1][i] = np.logaddexp(Q_hat[j-1][i], Q_hat[j][i] + unpaired_sc)

#             if i >= 2:
#                 paired_sc = float(NEG_INF) # calculate weighted paired score
#                 for c1, c2 in _allowed_pairs:
#                     paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][c1]) + np.log(X[j][c2]) + (-paired(c1, c2) / RT))

#                 for k in Q[i-2]:
#                     Q_hat[i-2][k] = np.logaddexp(Q_hat[i-2][k], Q_hat[j][k] + Q[j-1][i] + paired_sc)
#                     Q_hat[j-1][i] = np.logaddexp(Q_hat[j-1][i], Q_hat[j][k] + Q[i-2][k] + paired_sc)
#                     # grad[j][i-1] = np.logaddexp(grad[j][i-1], Q_hat[j][k] + Q[i-2][k] + Q[j-1][i])

#     return Q_hat, grad

