import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

kT = 61.63207755

def graph_mode2(n, y="", targeted=False):
    if y != "" and not targeted:
        file_name = f'n{n}_y'
    elif y != "" and targeted:
        file_name = f'n{n}_y_target'
    else:
        file_name = f'n{n}'

    pre_path = '.'
    if n == 12 and y == "":
        pre_path = '/scratch/tangwe'

    with open(f"{pre_path}/Qx/{file_name}.txt", 'r') as file:
        lines = file.read().split('\n')[:-1]
        if y != "": lines = lines[1:]

        val = []
        for line in lines:
            val.append(float(line.split(' ')[1]))

        val_log = np.array(val) * 100 / -kT
        val = np.exp(val_log)

        E_Qx = np.mean(val)

        # exact value
        E_log_Qx = np.mean(val_log)

        # sampling approx
        sampling = np.mean(np.random.choice(val_log, size=1000, replace=True))

        # first order approx: log E[Q(x)]
        first_order = np.log(E_Qx)

        # second order approx: log(E[Q(x)]) - Var(Q(x)) / (2 * E[Q(x)]^2)
        std_Qx = np.std(val)
        second_order = np.log(E_Qx) - ((std_Qx * std_Qx) / (2 * (E_Qx * E_Qx)))

        # Create scatter plot
        fig, ax1 = plt.subplots()
        ax1.hist(val, bins=50, alpha=0.3, color="orange", edgecolor='black')
        ax1.set_ylabel('Frequency', color="orange")
        ax1.set_xlabel('Q(x)')

        ax2 = ax1.twinx()
        max_val = max(val_log)
        min_val = min(val_log)
        gap = 0.06 * (max_val - min_val)
        ax2.set_ylim(min_val - gap, max_val + gap)
        ax2.scatter(val, val_log, marker='x', alpha=0.6)
        ax2.set_ylabel('log Q(x)', color="blue")
        ax2.set_xlabel('Q(x)')
        
        ax2.axhline(first_order, color='orange', alpha=0.8, linestyle='dashed', linewidth=1, label=f'Jensen: {first_order:.3f}')
        ax2.axhline(second_order, color='purple', alpha=0.8, linestyle='dashed', linewidth=1, label=f'Second Order: {second_order:.3f}')
        ax2.axhline(sampling, color='pink', alpha=0.8, linestyle='dashed', linewidth=1, label=f'Sampling: {sampling:.3f}')
        ax2.axhline(E_log_Qx, color='red', alpha=0.8, linewidth=1, label=f'E[log Q(x)]: {E_log_Qx:.3f}')
        # ax2.legend()
        
        ax2.plot([], [], ' ', label=' ')

        ax2.axvline(E_Qx, color='blue', alpha=0.8, linewidth=1, label=f'E[Q(x)]: {E_Qx:.2e}')
        ax2.axvline(E_Qx + std_Qx, color='green', alpha=0.8, linestyle='dashed', linewidth=1, label=f'1 STDs, std = {std_Qx:.2e}')
        ax2.axvline(E_Qx - std_Qx, color='green', alpha=0.8, linestyle='dashed', linewidth=1)
        ax2.legend()

        # Add labels and title
        if y == "":
            plt.title(f'log Q(x) vs. Q(x), Uniform, n = {n}')
        elif targeted:
            plt.title(f'log Q(x) vs. Q(x), Targeted, n = {n}, y = {y}')
        else:
            plt.title(f'log Q(x) vs. Q(x), Uniform, n = {n}, y = {y}')

        # Show plot
        # plt.show()
        plt.tight_layout()
        plt.savefig(f'Qx_plot2/{file_name}.png', format='png', bbox_inches='tight')


def graph_mode1(n, y=""):
    if y != "":
        file_name = f'n{n}_y'
    else:
        file_name = f'n{n}'

    with open(f"Qx/{file_name}.txt", 'r') as file:
        lines = file.read().split('\n')
        
        val = []
        for line in lines:
            if len(line) > 0:
                val.append(float(line.split(' ')[1]))
                
        mean_value = np.mean(val)
        std_dev = np.std(val)

        print(f"mean: {mean_value}")

        # Set the number of standard deviations away from the mean

        # Calculate the positions for the standard deviation lines

        plt.hist(val, bins=50, edgecolor='black')
        plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.4f}')

        for num_std_dev in range(1, 4):  # Change this value as needed
            lower_bound = mean_value - num_std_dev * std_dev
            upper_bound = mean_value + num_std_dev * std_dev
            plt.axvline(lower_bound, color='green', linestyle='dashed', linewidth=1, label=f'{num_std_dev} STDs, std = {std_dev:.4f}')
            plt.axvline(upper_bound, color='green', linestyle='dashed', linewidth=1)

        plt.title(f'Distribution of log Q(x) * -kT / 100.0, n = {n}')
        # plt.title(f'Distribution of log Q(x) * -kT / 100.0, n = {n}, y = (((...)))')
        if y != "":
            plt.title(f'Distribution of log Q(x) * -kT / 100.0, n = {n}, y={y}')
        else:
            plt.title(f'Distribution of log Q(x) * -kT / 100.0, n = {n}')

        plt.xlabel('Free Energy of Ensemble')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'Qx_plot/{file_name}.png', format='png', bbox_inches='tight')

parser = argparse.ArgumentParser()
parser.add_argument("--y", type=str, default="")
parser.add_argument("--n", type=int, default=0)
parser.add_argument("--targeted", action='store_true', default=False)
parser.add_argument("--mode", type=int, default=1)

args = parser.parse_args()

n = args.n
y = ""

if args.y != "":
    y = args.y
    n = len(y)

if args.mode == 1:
    if y == "" and n > 0:
        graph_mode1(n)
    else:
        graph_mode1(n, y)
elif args.mode == 2:
    if y == "" and n > 0:
        graph_mode2(n)
    else:
        graph_mode2(n, y, args.targeted)