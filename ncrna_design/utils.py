import numpy as np

A,C,G,U = range(4)

class Mode:
    def __init__(self, learning_rate=0.01, num_steps=2000, sharpturn=3, penalty=10, coupled=True, test=False, initialization='uniform', objective='pyx_jensen') -> None:
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
            elif mode.init == 'targeted':
                if np.random.randint(2):
                    params[i, j] = np.array([0.49, 0.51, 0., 0., 0., 0.])
                else:
                    params[i, j] = np.array([0.51, 0.49, 0., 0., 0., 0.])
        else:
            if mode.init == 'uniform' or mode.init == 'targeted': # j unpaired
                # params[j, j] = {'A': .25, 'C': .25, 'G': .25, 'U': .25}
                params[j, j] = np.array([.25, .25, .25, .25])
    
    return params

def print_distribution(dist, mode, file=None):
    if file:
        if mode.coupled:
            for idx, prob in dist.items():
                file.write(f"{idx[0]} {idx[1]}, {prob}\n")
        else:
            for i, x in enumerate(dist):
                file.write(f"{i:2d}: A {x[A]:.2f}, C {x[C]:.2f}, G {x[G]:.2f}, U {x[U]:.2f}\n")
    else:
        if mode.coupled:
            for idx, prob in dist.items():
                print(f"{idx[0]} {idx[1]}, {prob}")
        else:
            for i, x in enumerate(dist):
                print(f"{i:2d}: A {x[A]:.2f}, C {x[C]:.2f}, G {x[G]:.2f}, U {x[U]:.2f}")