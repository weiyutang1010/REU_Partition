import sys
import matplotlib.pyplot as plt

if sys.argv[2] == 'samples':
    with open(f'samples/{sys.argv[1]}.txt', 'r') as file:
        lines = file.read().split('\n')
        
        n = int(lines[0].split(', ')[0].split(': ')[1])
        exact_value = float(lines[n+1].split(': ')[1])
        jensen_value_simple = float(lines[n+2].split(': ')[1])
        jensen_value_full = float(lines[n+3].split(': ')[1])

        k = []
        val = []
        for line in lines[n+4:n+2004]:
            if len(line) > 0:
                k.append(int(line.split()[0]))
                val.append(float(line.split()[1]))

        plt.ylim(exact_value-0.05, exact_value+0.05)

        plt.axhline(y=exact_value, color='r', label='Exact Value (Full model)')
        plt.axhline(y=jensen_value_simple, color='orange', label='Jensen Approximation (Simple model)')
        plt.axhline(y=jensen_value_full, color='g', label='Jensen Approximation (Full model)')

        plt.xlabel('Number of Samples')
        plt.ylabel('E[log Q(x)] (kcal/mol)')
        plt.title(f'Monte Carlo Sampling, n = {n}, Uniform Distribution')

        plt.plot(k, val, label='Sample Approximation (Full model)')
        plt.legend()
        plt.savefig(f'samples/{sys.argv[1]}.png', format='png', bbox_inches='tight')
else:
    with open(f'approx_gap/{sys.argv[1]}.txt', 'r') as file:
        lines = file.read().split('\n')
        
        entropy = []
        gap = []
        n = int(lines[0].split(', ')[0].split(': ')[1])
        for line in lines[2:200]:
            if len(line) > 0:
                entropy.append(float(line.split(", ")[0]))
                gap.append(float(line.split(", ")[1]))

        plt.xlabel('Entropy')
        plt.ylabel('Exact - Approximation')
        plt.title(f'Approximation Gap vs. Avg Positional Entropy, n = {n}')

        plt.scatter(entropy, gap, label='k = 2000', marker='x', alpha=0.6)
        plt.legend()
        plt.savefig(f'approx_gap/{sys.argv[1]}.png', format='png', bbox_inches='tight')