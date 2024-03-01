import time
import numpy as np
from collections import defaultdict

from utils import generate_sequences, marginalize
from utils import read_sequences_Dy, get_probability_Dy, get_gradient_Dy

RT = 1.
nucs = 'ACGU'
nucpairs = ['CG', 'GC', 'AU', 'UA', 'GU', 'UG']
_allowed_pairs = [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)] 
_invalid_pairs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)]
NEG_INF = float('-1e18')
SMALL_NUM = float('1e-18')
CG,GC,AU,UA,GU,UG = range(6)
A,C,G,U = range(4)

sequences = []
log_Q_cached = {}

def paired(c1, c2, mode):
    _allowed_pairs_idx = {(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    _allowed_pairs = {"CG": -3, "GC": -3, "AU": -2, "UA":-2, "GU": -1, "UG":-1}

    if (c1, c2) in _allowed_pairs_idx:
        c1, c2 = nucs[c1], nucs[c2]

    if c1 + c2 in _allowed_pairs:
        return _allowed_pairs[c1 + c2]
    else:
        return mode.penalty

def unpaired(c='A'):
    return 1. # for now set every unpaired to same score

def objective(rna_struct, dist, mode):
    n = len(rna_struct)

    X = np.array([[.0, .0, .0, .0] for i in range(n)])
    if mode.coupled:
        marginalize(dist, X)
    else:
        X = dist

    objective_value, gradient = 0.0, []
    if mode.obj == 'pyx_jensen':
        # E[Delta G(x, y)] + log E[Q(x)]
        log_Q_hat = expected_inside_partition_log(X, mode)
        grad1 = expected_outside_partition_log(log_Q_hat, X, mode)
        Delta_G, grad2 = expected_free_energy(rna_struct, X, mode)
        
        objective_value = Delta_G + log_Q_hat[n-1][0]
        gradient = grad1 + grad2
    elif mode.obj == 'pyx_noapprox':
        # E[Delta G(x, y)] + E[log Q(x)]
        log_Q_hat, grad1 = E_log_Q(X, mode)
        Delta_G, grad2 = expected_free_energy(rna_struct, X, mode)
        
        objective_value = Delta_G + log_Q_hat
        gradient = grad1 + grad2
    elif mode.obj == 'pyx_max':
        # - log E[e^{-Delta G(x, y)}] + log E[Q(x)]
        log_Q_hat = expected_inside_partition_log(X, mode)
        grad1 = expected_outside_partition_log(log_Q_hat, X, mode)
        Delta_G, grad2 = expected_free_energy(rna_struct, X, mode)

        objective_value = - Delta_G + log_Q_hat[n-1][0]
        gradient = grad1 + grad2

    elif mode.obj == 'deltaG':
        Delta_G, grad2 = expected_free_energy(rna_struct, X, mode)
        objective_value = Delta_G

    elif mode.obj == 'pyx_jensen_Dy':
        # E[Delta G(D_y, y)] + log E[Q(D_y)]
        log_Q_hat = expected_inside_partition_log_Dy(rna_struct, dist, X, mode)
        grad1 = expected_outside_partition_log_Dy(rna_struct, log_Q_hat, dist, X, mode)
        Delta_G, grad2 = expected_free_energy_Dy(rna_struct, dist, mode)

        objective_value = Delta_G + log_Q_hat[n-1][0]
        
        gradient = {}
        for idx in grad1:
            gradient[idx] = grad1[idx] + grad2[idx]
    elif mode.obj == 'pyx_noapprox_Dy':
        # E[Delta G(x, y)] + E[log Q(x)]
        # Only works for (((...))) and ((((...)))). otherwise need to generate new Qx file
        log_Q_hat, grad1 = E_log_Q_Dy(rna_struct, dist, mode)
        Delta_G, grad2 = expected_free_energy_Dy(rna_struct, dist, mode)
        
        objective_value = Delta_G + log_Q_hat
        gradient = {}
        for idx in grad1:
            gradient[idx] = grad1[idx] + grad2[idx]
    elif mode.obj == 'pyx_sampling_Dy':
        # TODO: add mode: nussinov or vienna
        log_Q_hat, grad1 = E_log_Q_Dy_sampling(rna_struct, dist, mode)
        Delta_G, grad2 = expected_free_energy_Dy(rna_struct, dist, mode) #TODO: create vienna version
        
        objective_value = Delta_G + log_Q_hat
        gradient = {}
        for idx in grad1:
            gradient[idx] = grad1[idx] + grad2[idx]
    else:
        print("Error: Objective Not Found")
        exit(1)

    
    return objective_value, gradient, grad1, grad2

def expected_inside_partition_log(X, mode):
    """
        log E[Q(x)]
    """
    n = len(X)
    Q_hat = defaultdict(lambda: defaultdict(lambda: NEG_INF))

    for j in range(n):
        Q_hat[j-1][j] = 0.

    for j in range(n):
        for i in Q_hat[j-1]:
            unpaired_sc = NEG_INF
            for nucj in range(4):
                unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j][nucj] + SMALL_NUM) + (-unpaired(nucj) / RT))
            Q_hat[j][i] = np.logaddexp(Q_hat[j][i], Q_hat[j-1][i] + unpaired_sc)

            if i > 0 and j-(i-1) > mode.sharpturn:
                paired_sc = NEG_INF # calculate weighted paired score
                for nuci_1, nucj in _allowed_pairs:
                    paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][nuci_1] + SMALL_NUM) + np.log(X[j][nucj] + SMALL_NUM) + (-paired(nuci_1, nucj, mode) / RT))
                
                for k in Q_hat[i-2]:
                    Q_hat[j][k] = np.logaddexp(Q_hat[j][k], Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc)

    return Q_hat


def expected_outside_partition_log(Q_hat, X, mode):
    """
    log E[Q(x)]
    Backpropagation of Expected Inside Partition. Q_hat, O_hat, and gradient are calculated in log space.
       The dynet result is given by exp(gradient)."""
    n = len(X)
    O_hat = defaultdict(lambda: defaultdict(lambda: NEG_INF))
    gradient = np.array([[NEG_INF, NEG_INF, NEG_INF, NEG_INF] for _ in range(n)])

    O_hat[n-1][0] = 0.
    for j in range(n-1, -1, -1):
        for i in Q_hat[j-1]:
            unpaired_sc = NEG_INF # calculate weighted unpaired score
            for nucj in range(4):
                unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j][nucj] + SMALL_NUM) + (-unpaired(nucj) / RT))
            O_hat[j-1][i] = np.logaddexp(O_hat[j-1][i], O_hat[j][i] + Q_hat[j-1][i] + unpaired_sc - Q_hat[j][i])

            for nucj in range(4):
                # gradient of Q_hat[1][n] with respect to unpaired_sc
                grad = O_hat[j][i] + Q_hat[j-1][i] + unpaired_sc - Q_hat[j][i]
                # gradient of unpaired_sc with respect to nucj
                softmax = np.log(X[j][nucj] + SMALL_NUM) + (-unpaired(nucj) / RT) - unpaired_sc
                gradient[j][nucj] =  np.logaddexp(gradient[j][nucj], grad - np.log(X[j][nucj] + SMALL_NUM) + softmax)

            if i > 0 and j-(i-1) > mode.sharpturn:
                paired_sc = NEG_INF # calculate weighted paired score
                for c1, c2 in _allowed_pairs:
                    paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][c1] + SMALL_NUM) + np.log(X[j][c2] + SMALL_NUM) + (-paired(c1, c2, mode) / RT))

                for k in Q_hat[i-2]:
                    O_hat[i-2][k] = np.logaddexp(O_hat[i-2][k], O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k])
                    O_hat[j-1][i] = np.logaddexp(O_hat[j-1][i], O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k])

                    # gradient of Q_hat[1][n] with respect to paired_sc
                    grad = O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k]

                    # gradient of Q_hat[1][n] with respect to nuci_1, nucj
                    for nuci_1, nucj in _allowed_pairs:
                        softmax = np.log(X[i-1][nuci_1] + SMALL_NUM) + np.log(X[j][nucj] + SMALL_NUM) + (-paired(nuci_1, nucj, mode)) - paired_sc
                        gradient[i-1][nuci_1] = np.logaddexp(gradient[i-1][nuci_1], grad - np.log(X[i-1][nuci_1] + SMALL_NUM) + softmax)
                        gradient[j][nucj] = np.logaddexp(gradient[j][nucj], grad - np.log(X[j][nucj] + SMALL_NUM) + softmax)

    return np.exp(gradient)

def expected_free_energy(struct, X, mode):
    """E[Delta G(x, y)]"""
    n = len(struct)
    free_energy = 0.

    gradient = np.array([[0., 0., 0., 0.] for _ in range(n)])

    stack = []
    for j in range(n):
        if struct[j] == '(':
            stack.append(j)
        elif struct[j] == ')':
            i = stack.pop()

            score_ij = 0.
            
            for nuci, nucj in _allowed_pairs + _invalid_pairs:
                score_ij += X[i][nuci] * X[j][nucj] * paired(nuci, nucj, mode)
                gradient[i][nuci] += X[j][nucj] * paired(nuci, nucj, mode)
                gradient[j][nucj] += X[i][nuci] * paired(nuci, nucj, mode)

            free_energy += score_ij
        else:
            free_energy += unpaired()

    return free_energy, gradient

def expected_free_energy_Dy(struct, dist, mode):
    """E[Delta G(x, y)]"""
    n = len(struct)
    free_energy = 0.

    gradient = defaultdict(lambda: np.array([0., 0., 0., 0.]))

    stack = []
    for j in range(n):
        if struct[j] == '(':
            stack.append(j)
        elif struct[j] == ')':
            i = stack.pop()

            score_ij = 0.
            
            gradient[i, j] = np.array([0.] * 6)
            for nuci, nucj in _allowed_pairs:
                score_ij += dist[i, j][pairs(nuci, nucj)] * paired(nuci, nucj, mode)
                gradient[i, j][pairs(nuci, nucj)] += paired(nuci, nucj, mode)

            free_energy += score_ij
        else:
            free_energy += unpaired()

    return free_energy, gradient

def log_Q(x, mode):
    _allowed_pairs = [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2), ('C', 'G'), ('G', 'C'), ('A', 'U'), ('U', 'A'), ('G', 'U'), ('U', 'G')] 
    if mode.energy_model == 'nussinov':
        n = len(x)
        Q = defaultdict(lambda: defaultdict(lambda: NEG_INF))

        for j in range(n):
            Q[j-1][j] = 0.

        for j in range(n):
            for i in Q[j-1]:
                Q[j][i] = np.logaddexp(Q[j][i], Q[j-1][i] + (-unpaired(x[j])))

                if i > 0 and j - (i-1) > mode.sharpturn and (x[i-1], x[j]) in _allowed_pairs:
                    for k in Q[i-2]:
                        Q[j][k] = np.logaddexp(Q[j][k], Q[i-2][k] + Q[j-1][i] + (-paired(x[i-1], x[j], mode))) 

        return Q[n-1][0]
    else:
        # TODO: calls LinearPartition
        return 0.

def E_log_Q(X, mode):
    n = len(X)
    obj = 0.
    grad = np.array([[0., 0., 0., 0.] for _ in range(n)])

    if n not in sequences:
        # Generate all 4^n sequences
        sequences[n] = generate_sequences(range(4), n=n)

    for seq in sequences[n]:
        prob = 1.
        for j, nuc in enumerate(seq):
            prob *= X[j][nuc]

        log_Qx = log_Q(seq, mode)
        obj += prob * log_Qx

        prob_grad = [1. for _ in range(n)]
        # Compute products to the left of each element
        for j in range(1, n):
            prob_grad[j] *= X[j-1][seq[j-1]] * prob_grad[j-1]

        # Compute products to the right of each element
        right_prod = 1.
        for j in range(n - 2, -1, -1):
            right_prod *= X[j+1][seq[j+1]]
            prob_grad[j] *= right_prod

        for j, nuc in enumerate(seq):
            grad[j][nuc] += prob_grad[j] * log_Qx

    return obj, grad

def E_log_Q_Dy(rna_struct, dist, mode):
    n = len(rna_struct)
    obj = 0.

    gradient = {}
    for idx, probs in dist.items():
        # initialization
        i, j = idx
        if i == j:
            gradient[i, j] = np.array([0., 0., 0., 0.]) 
        else:
            gradient[i, j] = np.array([0., 0., 0., 0., 0., 0.])

    global sequences, log_Q_cached
    if len(sequences) == 0:
        # Read all 6^pairs * 4^unpairs sequences
        sequences, log_Q_cached = read_sequences_Dy(n)

    for seq in sequences:
        obj += get_probability_Dy(dist, seq) * log_Q_cached[seq]
        grad = get_gradient_Dy(dist, seq)

        for idx, probs in dist.items():
            gradient[idx] += grad[idx] * log_Q_cached[seq]

    return obj, gradient

def E_log_Q_Dy_sampling(rna_struct, dist, mode):
    n = len(rna_struct)
    obj = 0.

    gradient = {}
    for idx, probs in dist.items():
        # initialization
        i, j = idx
        if i == j:
            gradient[i, j] = np.array([0., 0., 0., 0.]) 
        else:
            gradient[i, j] = np.array([0., 0., 0., 0., 0., 0.])

    # sample k sequences from the distribution
    samples = []
    seq = ['A'] * n
    for _ in range(mode.sample_size):
        for idx, probs in dist.items():
            i, j = idx
            if i == j:
                seq[j] = np.random.choice(['A','C','G','U'], p=probs)
            else:
                rand_pair = np.random.choice(['CG','GC','AU','UA', 'GU', 'UG'], p=probs)
                seq[i], seq[j] = rand_pair[0], rand_pair[1]
        samples.append(seq[:])

    # TODO: can run in parallel
    obj = 0.
    for seq in samples:
        log_Qx = log_Q(seq, mode)
        obj += log_Qx

        grad = get_gradient_Dy(dist, seq)
        prob = get_probability_Dy(dist, seq)
        for idx, probs in grad.items():
            gradient[idx] += log_Qx * (grad[idx] / prob) 

    for idx in gradient.keys():
        gradient[idx] /= mode.sample_size
    
    obj /= mode.sample_size

    return obj, gradient

def E_exp_DeltaG(X, mode):
    """
        - log E[e^{-Delta G(x, y)}]
        not done yet
    """
    n = len(X)
    obj = 0.
    # grad = np.array([[0., 0., 0., 0.] for _ in range(n)])
    grad = np.array([[0., 0., 0., 0.] for _ in range(n)])

    if n not in sequences:
        # Generate all 4^n sequences
        sequences[n] = generate_sequences(range(4), n=n)

    for seq in sequences[n]:
        prob = 1.
        for j, nuc in enumerate(seq):
            prob *= X[j][nuc]

        log_Qx = log_Q(seq, mode)
        obj += prob * log_Qx

        prob_grad = [1. for _ in range(n)]
        # Compute products to the left of each element
        for j in range(1, n):
            prob_grad[j] *= X[j-1][seq[j-1]] * prob_grad[j-1]

        # Compute products to the right of each element
        right_prod = 1.
        for j in range(n - 2, -1, -1):
            right_prod *= X[j+1][seq[j+1]]
            prob_grad[j] *= right_prod

        for j, nuc in enumerate(seq):
            grad[j][nuc] += prob_grad[j] * log_Qx

    return obj, grad

def pairs(a, b):
    table = {(1, 2): 0, (2, 1): 1, (0, 3): 2, (3, 0): 3, (2, 3): 4, (3, 2): 5}
    return table[a, b]

def expected_inside_partition_log_Dy(rna_struct, dist, X, mode):
    """
        log E[Q(D_y)]
    """
    n = len(rna_struct)
    Q_hat = defaultdict(lambda: defaultdict(lambda: NEG_INF))

    stack = []
    pairs_idx = set()
    for j, c in enumerate(rna_struct):
        if c == '(': stack.append(j)
        elif c == ')': pairs_idx.add((stack.pop(), j))

    X = [0 for _ in range(n)]
    marginalize(dist, X)

    for j in range(n):
        Q_hat[j-1][j] = 0.

    for j in range(n):
        for i in Q_hat[j-1]:
            unpaired_sc = NEG_INF
            for nucj in range(4):
                unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j][nucj] + SMALL_NUM) + (-unpaired(nucj) / RT))
            Q_hat[j][i] = np.logaddexp(Q_hat[j][i], Q_hat[j-1][i] + unpaired_sc)

            if i > 0 and j-(i-1) > mode.sharpturn:
                paired_sc = NEG_INF # calculate weighted paired score
                if (i-1, j) in pairs_idx:
                    for nuci_1, nucj in _allowed_pairs:
                        paired_sc = np.logaddexp(paired_sc, np.log(dist[i-1, j][pairs(nuci_1, nucj)] + SMALL_NUM) + (-paired(nuci_1, nucj, mode) / RT))
                else:
                    for nuci_1, nucj in _allowed_pairs:
                        paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][nuci_1] + SMALL_NUM) + np.log(X[j][nucj] + SMALL_NUM) + (-paired(nuci_1, nucj, mode) / RT))
                
                for k in Q_hat[i-2]:
                    Q_hat[j][k] = np.logaddexp(Q_hat[j][k], Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc)

    return Q_hat

def expected_outside_partition_log_Dy(rna_struct, Q_hat, dist, X, mode):
    """
    log E[Q(x)]
    Backpropagation of Expected Inside Partition. Q_hat, O_hat, and gradient are calculated in log space.
       The dynet result is given by exp(gradient)."""
    n = len(rna_struct)
    O_hat = defaultdict(lambda: defaultdict(lambda: NEG_INF))

    gradient = np.array([[NEG_INF, NEG_INF, NEG_INF, NEG_INF] for _ in range(n)])
    gradient_pairs = {}

    stack = []
    for j, c in enumerate(rna_struct):
        if c == '(': stack.append(j)
        elif c == ')': gradient_pairs[stack.pop(), j] = np.array([NEG_INF] * 6)
        else: gradient_pairs[j, j] = np.array([NEG_INF] * 4)

    stack = []
    pairs_idx = set()
    for j, c in enumerate(rna_struct):
        if c == '(': stack.append(j)
        elif c == ')': pairs_idx.add((stack.pop(), j))

    O_hat[n-1][0] = 0.
    for j in range(n-1, -1, -1):
        for i in Q_hat[j-1]:
            unpaired_sc = NEG_INF # calculate weighted unpaired score
            for nucj in range(4):
                unpaired_sc = np.logaddexp(unpaired_sc, np.log(X[j][nucj] + SMALL_NUM) + (-unpaired(nucj) / RT))
            O_hat[j-1][i] = np.logaddexp(O_hat[j-1][i], O_hat[j][i] + Q_hat[j-1][i] + unpaired_sc - Q_hat[j][i])

            for nucj in range(4):
                # gradient of Q_hat[1][n] with respect to unpaired_sc
                grad = O_hat[j][i] + Q_hat[j-1][i] + unpaired_sc - Q_hat[j][i]
                # gradient of unpaired_sc with respect to nucj
                softmax = np.log(X[j][nucj] + SMALL_NUM) + (-unpaired(nucj) / RT) - unpaired_sc
                gradient[j][nucj] =  np.logaddexp(gradient[j][nucj], grad - np.log(X[j][nucj] + SMALL_NUM) + softmax)

            if i > 0 and j-(i-1) > mode.sharpturn:
                paired_sc = NEG_INF # calculate weighted paired score
                if (i-1, j) in pairs_idx:
                    for c1, c2 in _allowed_pairs:
                        paired_sc = np.logaddexp(paired_sc, np.log(dist[i-1, j][pairs(c1, c2)] + SMALL_NUM) + (-paired(c1, c2, mode) / RT))

                    for k in Q_hat[i-2]:
                        O_hat[i-2][k] = np.logaddexp(O_hat[i-2][k], O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k])
                        O_hat[j-1][i] = np.logaddexp(O_hat[j-1][i], O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k])

                        # gradient of Q_hat[1][n] with respect to paired_sc
                        grad = O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k]

                        # gradient of Q_hat[1][n] with respect to nuci_1, nucj
                        for nuci_1, nucj in _allowed_pairs:
                            softmax = np.log(dist[i-1, j][pairs(nuci_1, nucj)] + SMALL_NUM) + (-paired(nuci_1, nucj, mode)) - paired_sc
                            gradient_pairs[i-1, j][pairs(nuci_1, nucj)] = np.logaddexp(gradient_pairs[i-1, j][pairs(nuci_1, nucj)], grad - np.log(dist[i-1, j][pairs(nuci_1, nucj)] + SMALL_NUM) + softmax)
                            # gradient[i-1][nuci_1] = np.logaddexp(gradient[i-1][nuci_1], grad - np.log(X[i-1][nuci_1] + SMALL_NUM) + softmax)
                            # gradient[j][nucj] = np.logaddexp(gradient[j][nucj], grad - np.log(X[j][nucj] + SMALL_NUM) + softmax)
                else:
                    for c1, c2 in _allowed_pairs:
                        paired_sc = np.logaddexp(paired_sc, np.log(X[i-1][c1] + SMALL_NUM) + np.log(X[j][c2] + SMALL_NUM) + (-paired(c1, c2, mode) / RT))

                    for k in Q_hat[i-2]:
                        O_hat[i-2][k] = np.logaddexp(O_hat[i-2][k], O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k])
                        O_hat[j-1][i] = np.logaddexp(O_hat[j-1][i], O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k])

                        # gradient of Q_hat[1][n] with respect to paired_sc
                        grad = O_hat[j][k] + Q_hat[i-2][k] + Q_hat[j-1][i] + paired_sc - Q_hat[j][k]

                        # gradient of Q_hat[1][n] with respect to nuci_1, nucj
                        for nuci_1, nucj in _allowed_pairs:
                            softmax = np.log(X[i-1][nuci_1] + SMALL_NUM) + np.log(X[j][nucj] + SMALL_NUM) + (-paired(nuci_1, nucj, mode)) - paired_sc
                            gradient[i-1][nuci_1] = np.logaddexp(gradient[i-1][nuci_1], grad - np.log(X[i-1][nuci_1] + SMALL_NUM) + softmax)
                            gradient[j][nucj] = np.logaddexp(gradient[j][nucj], grad - np.log(X[j][nucj] + SMALL_NUM) + softmax)

    for idx, grad in gradient_pairs.items():
        i, j = idx
        if i != j:
            gradient_pairs[idx] = np.exp(grad) + np.exp(np.array([gradient[i][C] + gradient[j][G],
                                                gradient[i][G] + gradient[j][C],
                                                gradient[i][A] + gradient[j][U],
                                                gradient[i][U] + gradient[j][A],
                                                gradient[i][G] + gradient[j][U],
                                                gradient[i][U] + gradient[j][G]]))
        else:
            gradient_pairs[idx] = np.exp(gradient[j])

    return gradient_pairs