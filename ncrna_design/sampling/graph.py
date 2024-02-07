import sys
import matplotlib.pyplot as plt

if sys.argv[2] == 'samples':
    with open(f'samples/{sys.argv[1]}.txt', 'r') as file:
        lines = file.read().split('\n')
        
        n = int(lines[0].split(', ')[0].split(': ')[1])
        exact_value = float(lines[n+1].split(':  ')[1])

        k = []
        val = []
        for line in lines[n+2:n+2002]:
            if len(line) > 0:
                k.append(int(line.split()[0]))
                val.append(float(line.split()[1]))

        plt.axhline(y=exact_value, color='r', label='Exact Value')

        plt.xlabel('Number of Samples')
        plt.ylabel('E[log Q(x)]')
        plt.title('Monte Carlo Sampling, n = 9, Uniform Distribution')

        plt.plot(k, val, label='Approximated Value')
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
        plt.ylabel('Approximation Gap')
        plt.title(r'$E[\log Q(x)] - \log E[Q(x)]$, n = 8')

        plt.scatter(entropy, gap, label='k = 1000', marker='x', alpha=0.6)
        plt.legend()
        plt.savefig(f'approx_gap/{sys.argv[1]}.png', format='png', bbox_inches='tight')