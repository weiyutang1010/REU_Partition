import sys
import matplotlib.pyplot as plt

if sys.argv[2] == 'samples':
    with open(f'samples/{sys.argv[1]}.txt', 'r') as file:
        lines = file.read().split('\n')
        
        n = int(lines[0].split(', ')[0].split(': ')[1])
        exact_value = float(lines[n+1].split(': ')[1])
        jensen_value_full = float(lines[n+2].split(': ')[1])

        k = []
        val = []
        for line in lines[n+4:n+2004]:
            if len(line) > 0:
                k.append(int(line.split()[0]))
                val.append(float(line.split()[1]))

        plt.ylim(exact_value-0.0002, exact_value+0.0002)

        plt.axhline(y=exact_value, color='r', alpha=1.0, label='Exact Value (Full model)')
        # plt.axhline(y=jensen_value_simple, color='orange', label='Jensen Approximation (Simple model)')
        plt.axhline(y=jensen_value_full, color='g', alpha=0.6, label='Jensen Approximation (Full model)')

        plt.xlabel('Number of Samples')
        plt.ylabel('Expected Free Energy of Ensemble (kcal/mol)')
        plt.title(f'Monte Carlo Sampling, n = {n}, Uniform Distribution')

        plt.plot(k, val, label='Sample Approximation (Full model)')
        plt.legend()
        plt.savefig(f'samples/{sys.argv[1]}.png', format='png', bbox_inches='tight')
elif sys.argv[2] == 'approx':
    with open(f'approx_gap/{sys.argv[1]}.txt', 'r') as file:
        lines = file.read().split('\n')

        
        entropy = []
        sampling = []
        jensen = []
        exact = []
        n = int(lines[0].split(', ')[0].split(': ')[1])
        k = int(lines[0].split(', ')[1].split(': ')[1])
        for line in lines[2:]:
            if len(line) > 0:
                entropy.append(float(line.split(", ")[0]))
                sampling.append(float(line.split(", ")[1]))
                jensen.append(float(line.split(", ")[2]))
                exact.append(float(line.split(", ")[3]))

        sampling_gap = [x - y for x, y in zip(sampling, exact)]
        jensen_gap = [x - y for x, y in zip(jensen, exact)]

        plt.xlim(-0.0, 2.0)
        # plt.ylim(-0.025, 0.01)

        plt.xlabel('Avg Positional Entropy')
        plt.ylabel('Approximation Gap (Approx - Exact)')
        plt.title(f'Approximation Gap vs. Avg Positional Entropy, n = {n}')

        plt.axhline(y=0.0, color='r')
        plt.scatter(entropy, sampling_gap, label=f'Sampling k = {k}', marker='x', alpha=0.5)
        plt.scatter(entropy, jensen_gap, label='Jensen\'s Approx', marker='x', alpha=0.5)
        plt.legend()
        plt.savefig(f'approx_gap/{sys.argv[1]}.png', format='png', bbox_inches='tight')
if sys.argv[2] == 'samples_y':
    with open(f'samples_y/{sys.argv[1]}.txt', 'r') as file:
        lines = file.read().split('\n')
        
        n = int(lines[0].split(', ')[0].split(': ')[1])
        y = lines[0].split(', ')[2].split(': ')[1]
        exact_value = float(lines[1].split(': ')[1])
        jensen_value_full = float(lines[2].split(': ')[1])

        k = []
        val = []
        for line in lines[4:2005]:
            if len(line) > 0:
                k.append(int(line.split()[0]))
                val.append(float(line.split()[1]))

        plt.ylim(exact_value-2.2, exact_value+0.05)

        plt.axhline(y=exact_value, color='r', alpha=1.0, label='Exact Value (Full model)')
        # plt.axhline(y=jensen_value_simple, color='orange', label='Jensen Approximation (Simple model)')
        plt.axhline(y=jensen_value_full, color='g', alpha=0.6, label='Jensen Approximation (Full model)')

        plt.xlabel('Number of Samples')
        plt.ylabel('Expected Free Energy of Ensemble (kcal/mol)')
        plt.title(f'struct: {y}, n = {n}, Uniform Distribution')

        plt.plot(k, val, label='Sample Approximation (Full model)')
        plt.legend()
        plt.savefig(f'samples_y/{sys.argv[1]}.png', format='png', bbox_inches='tight')
elif sys.argv[2] == 'approx_y':
    with open(f'approx_gap_y/{sys.argv[1]}.txt', 'r') as file:
        lines = file.read().split('\n')
        
        entropy = []
        sampling = []
        jensen = []
        exact = []
        n = int(lines[0].split(', ')[0].split(': ')[1])
        k = int(lines[0].split(', ')[1].split(': ')[1])
        y = lines[0].split(', ')[3].split(': ')[1]
        for line in lines[2:]:
            if len(line) > 0:
                entropy.append(float(line.split(", ")[0]))
                sampling.append(float(line.split(", ")[1]))
                jensen.append(float(line.split(", ")[2]))
                exact.append(float(line.split(", ")[3]))

        sampling_gap = [x - y for x, y in zip(sampling, exact)]
        jensen_gap = [x - y for x, y in zip(jensen, exact)]

        # plt.ylim(-0.025, 0.01)

        plt.xlabel('Avg Positional Entropy')
        plt.ylabel('Approximation Gap (Approx - Exact)')
        plt.title(f'Approximation Gap vs. Avg Positional Entropy, y: {y}')

        plt.axhline(y=0.0, color='r')
        plt.scatter(entropy, sampling_gap, label=f'Sampling k = {k}', marker='x', alpha=0.5)
        plt.scatter(entropy, jensen_gap, label='Jensen\'s Approx', marker='x', alpha=0.5)
        plt.legend()
        plt.savefig(f'approx_gap_y/{sys.argv[1]}.png', format='png', bbox_inches='tight')