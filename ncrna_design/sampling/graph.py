import sys
import matplotlib.pyplot as plt

with open(f'{sys.argv[1]}.txt', 'r') as file:
    lines = file.read().split('\n')
    
    n = int(lines[0].split(', ')[0].split(': ')[1])
    exact_value = float(lines[n+1].split(':  ')[1])

    k = []
    val = []
    for line in lines[n+2:]:
        if len(line) > 0:
            k.append(int(line.split()[0]))
            val.append(float(line.split()[1]))

    plt.axhline(y=exact_value, color='r', label='Exact Value')

    plt.xlabel('Number of Samples')
    plt.ylabel('E[log Q(x)]')
    plt.title('Monte Carlo Sampling, n = 8, Uniform Distribution')

    plt.plot(k, val, label='Approximated Value')
    plt.legend()
    plt.savefig(f'{sys.argv[1]}.png', format='png', bbox_inches='tight')