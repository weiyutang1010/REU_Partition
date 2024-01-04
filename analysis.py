import os, sys
import numpy as np
from inside_outside import inside_prob
from score import paired, unpaired
from util import RNA

def free_energy(rna_seq, rna_struct):
    n = len(rna_seq)
    stack = []
    score = 0.

    for j, c in enumerate(rna_struct):
        if c == '(':
            stack.append(j)
        elif c == ')':
            i = stack.pop()
            score += paired(rna_seq[i], rna_seq[j])
        else:
            score += unpaired(c)

    return score

def get_boltz_prob(rna_seq, rna_struct):
    n = len(rna_seq)
    Q = inside_prob(RNA(rna_seq))
    delta_G =  free_energy(rna_seq, rna_struct)

    return np.exp(-delta_G) / Q[n][1]

def process_result_file(rna_id, result_file, baseline_file):
    lines = result_file.read().split('\n')
    b_lines = baseline_file.read().split('\n') # b for baseline (comparison)
    n, rna_struct = len(lines[0]), lines[0]

    obj, b_obj = [], []
    seqs, b_seqs = [], []
    for line in lines:
        if line.startswith("step: "):
            obj.append(float(line.split(', ')[1].split(': ')[1]))
            seqs.append(line.split(', ')[2].split(': ')[1])

    for line in b_lines:
        if line.startswith("step: "):
            b_obj.append(float(line.split(', ')[1].split(': ')[1]))
            b_seqs.append(line.split(', ')[2].split(': ')[1])

    seq_idx = lines.index('Optimal Sequence') + 2
    print(f"id: {rna_id}")
    print(rna_struct)
    print(lines[seq_idx])

    test = get_boltz_prob(seqs[0], rna_struct)

    # fractional solutions
    initial, final = np.exp(-obj[0]), np.exp(-obj[-1])
    b_final = np.exp(-b_obj[-1])

    print("Fractional Solution")
    print(f"Initial: {initial:.5f}, Final: {final:5f}, Diff: {final-initial:.5f} (exp[inf p(y | x)])")
    print(f"Coupled: {final:5f}, No Coupled: {b_final:5f}, Diff: {final-b_final:.5f} (exp[inf p(y | x)])")


    # integral solutions
    initial, final = get_boltz_prob(seqs[0], rna_struct), get_boltz_prob(seqs[-1], rna_struct)
    b_final = get_boltz_prob(b_seqs[-1], rna_struct)

    print("Integral Solution")
    print(f"Initial: {initial:.5f}, Final: {final:5f}, Diff: {final-initial:.5f} (p(y | x))")
    print(f"Coupled: {final:5f}, No Coupled: {b_final:5f}, Diff: {final-b_final:.5f} (p(y | x))")
    print()

def main():
    results_path = f'./results/{sys.argv[1]}'

    baseline_path = ''
    if len(sys.argv) > 2:
        baseline_path = f'./results/{sys.argv[2]}'
    
    ids = []
    if os.path.exists(results_path):
        for folder_path, _, filenames in os.walk(results_path):
            for filename in filenames:
                rna_id, extension = os.path.splitext(filename)
                ids.append(int(rna_id))
        
        sort_by_len = [8, 1, 23, 26, 15, 30, 88, 41, 3, 11, 57, 66, 40, 65, 20, 10, 33, 47]
        # for rna_id in ids:
        for rna_id in sort_by_len:
            results_filepath = os.path.join(results_path, f'{rna_id}.txt')
            baseline_filepath = os.path.join(baseline_path, f'{rna_id}.txt')

            with open(results_filepath, 'r') as result_file:
                with open(baseline_filepath, 'r') as baseline_file:
                    process_result_file(rna_id, result_file, baseline_file)
    else:
        print('Folder does not exist.')


if __name__ == '__main__':
    main()