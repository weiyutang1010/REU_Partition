import numpy as np

A,C,G,U = range(4)
CG,GC,AU,UA,GU,UG = range(6)

class Mode:
    def __init__(self, learning_rate=0.001, num_steps=2000, sharpturn=3, penalty=10, coupled=True, test=False, initialization='uniform', objective='pyx_jensen') -> None:
        self.lr = learning_rate
        self.num_steps = num_steps
        self.sharpturn = sharpturn
        self.penalty = penalty
        self.coupled = coupled
        self.test = test
        self.init = initialization
        self.obj = objective

    def print(self, file=None):
        if file:
            file.write(f"Learning Rate: {self.lr}, Number of Steps: {self.num_steps}, Sharpturn: {self.sharpturn}, Penalty: {self.penalty}, Coupled: {self.coupled}\n")
            file.write(f"Initialization: {self.init}, Objective: {self.obj}\n")
        else:
            print(f"Learning Rate: {self.lr}, Number of Steps: {self.num_steps}, Sharpturn: {self.sharpturn}, Penalty: {self.penalty}, Coupled: {self.coupled}")
            print(f"Initialization: {self.init}, Objective: {self.obj}")

def params_init(rna_struct, mode):
    """Return n x 6 matrix, each row is probability of CG, GC, AU, UA, GU, UG"""
    n = len(rna_struct)

    if not mode.coupled:
        if mode.init == 'uniform':
            return np.array([[.25, .25, .25, .25] for _ in range(n)])

    params = {}
    stack = []

    for j, c in enumerate(rna_struct):
        if c == '(':
            stack.append(j)
        elif c == ')':
            i = stack.pop() # i, j paired

            if mode.init == 'uniform':
                # params[i, j] = {'CG': 1/6, 'GC': 1/6, 'AU': 1/6, 'UA': 1/6, 'GU': 1/6, 'UG': 1/6}
                params[i, j] = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
                # params[i, j] = np.array([0., 0., .5, .5, 0., 0.])
                # params[i, j] = np.array([.8, .2, 0., 0., 0., 0.])
            elif mode.init == 'targeted':
                if np.random.randint(2):
                    params[i, j] = np.array([0.49, 0.51, 0., 0., 0., 0.])
                else:
                    params[i, j] = np.array([0.51, 0.49, 0., 0., 0., 0.])
            elif mode.init == 'random':
                params[i, j] = np.random.dirichlet(np.ones(6))
        else:
            # if mode.init == 'uniform' or mode.init == 'targeted': # j unpaired
                # params[j, j] = {'A': .25, 'C': .25, 'G': .25, 'U': .25}
            params[j, j] = np.array([.25, .25, .25, .25])
    
    return params

def print_distribution(dist, mode, file=None):
    if file:
        if mode.coupled:
            for idx, x in sorted(dist.items()):
                i, j = idx
                if i == j:
                    file.write(f"{i:2d}: A {x[A]:.4f}, C {x[C]:.4f}, G {x[G]:.4f}, U {x[U]:.4f}\n")
                else:
                    file.write(f"({i}, {j}): CG {x[CG]:.4f} GC {x[GC]:.4f} AU {x[AU]:.4f} UA {x[UA]:.4f} GU {x[GU]:.4f} UG {x[UG]:.4f}\n")
        else:
            for i, x in enumerate(dist):
                file.write(f"{i:2d}: A {x[A]:.4f}, C {x[C]:.4f}, G {x[G]:.4f}, U {x[U]:.4f}\n")
    else:
        if mode.coupled:
            for idx, x in sorted(dist.items()):
                i, j = idx
                if i == j:
                    print(f"{i:2d}: A {x[A]:.4f}, C {x[C]:.4f}, G {x[G]:.4f}, U {x[U]:.4f}\n")
                else:
                    print(f"({i}, {j}): CG {x[CG]:.4f} GC {x[GC]:.4f} AU {x[AU]:.4f} UA {x[UA]:.4f} GU {x[GU]:.4f} UG {x[UG]:.4f}\n")
        else:
            for i, x in enumerate(dist):
                print(f"{i:2d}: A {x[A]:.4f}, C {x[C]:.4f}, G {x[G]:.4f}, U {x[U]:.4f}\n")

nucs = 'ACGU'
nucpairs = ['CG', 'GC', 'AU', 'UA', 'GU', 'UG']
_allowed_pairs = [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)] 
def get_intergral_solution(rna_struct, dist, n, mode):
    seq = ['A' for _ in range(n)]

    if mode.coupled:
        for idx, prob in dist.items():
            prob = [round(x, 4) for x in prob]
            i, j = idx
            if i == j:
                seq[i] = nucs[np.argmax(prob)]
            else:
                seq[i] = nucpairs[np.argmax(prob)][0]
                seq[j] = nucpairs[np.argmax(prob)][1]
    else:
        stack = []
        for j, c in enumerate(rna_struct):
            if c == '(':
                stack.append(j)
            elif c == ')':
                i = stack.pop()
                prob = -1.

                for nuci, nucj in _allowed_pairs:
                    if dist[i][nuci] * dist[j][nucj] > prob:
                        prob = dist[i][nuci] * dist[j][nucj]
                        seq[i] = nucs[nuci]
                        seq[j] = nucs[nucj]
            else:
                seq[j] = nucs[np.argmax(dist[j])]

    return "".join(seq)

def marginalize(params, X):
    """
        Create n x 4 marginalized probability distribution
        X[i] = [p(A), p(C), p(G), p(U)]
    """
    for idx, prob in params.items():
        i, j = idx
        if i == j:
            X[j] = prob
        else:
            # 0-CG, 1-GC, 2-AU, 3-UA, 4-GU, 5-UG
            X[i] = np.array([prob[AU], prob[CG], prob[GC] + prob[GU], prob[UA] + prob[UG]])
            X[j] = np.array([prob[UA], prob[GC], prob[CG] + prob[UG], prob[AU] + prob[GU]])

def generate_sequences(*args, n):
    pools = [tuple(pool) for pool in args] * n
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result
