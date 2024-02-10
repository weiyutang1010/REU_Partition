import sys
import numpy as np

kT = 61.63207755


n = sys.argv[1]
with open(f'Qx/n{n}.txt', 'r') as file:
    lines = file.read().split('\n')

    min_val = 1000.
    max_val = 0.
    for line in lines:
        if len(line) > 0:
            free_energy_ensemble = float(line.split(' ')[1])
            max_val = max(max_val, np.exp(free_energy_ensemble * 100.0 / -kT))
            min_val = min(min_val, np.exp(free_energy_ensemble * 100.0 / -kT))

    print(max_val, min_val)
