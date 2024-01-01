import os, sys
import numpy as np

def process_result_file(rna_id, result_file):
    lines = result_file.read().split('\n')
    n, rna_struct = len(lines[0]), lines[0]

    obj = []
    for line in lines:
        if line.startswith("step: "):
            obj.append(float(line.split(', ')[1].split(': ')[1]))


    seq_idx = lines.index('Optimal Sequence') + 2
    print(f"id: {rna_id}")
    print(rna_struct)
    print(lines[seq_idx])

    initial, final, diff = np.exp(-obj[0]), np.exp(-obj[-1]), np.exp(-obj[-1]) - np.exp(-obj[0])
    print(f"Initial: {initial:.5f}, Final: {final:5f}, Diff: {diff:.5f} (exp[inf p(y | x)])")
    print()

def main():
    results_path = f'./results/{sys.argv[1]}'
    
    ids = []
    if os.path.exists(results_path):
        for folder_path, _, filenames in os.walk(results_path):
            for filename in filenames:
                rna_id, extension = os.path.splitext(filename)
                ids.append(int(rna_id))
        
        ids.sort()
        for rna_id in ids:
            file_path = os.path.join(folder_path, f'{rna_id}.txt')
            with open(file_path, 'r') as result_file:
                process_result_file(rna_id, result_file)
    else:
        print('Folder does not exist.')


if __name__ == '__main__':
    main()