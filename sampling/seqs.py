#!/usr/bin/env python3

import sys
from itertools import product


# Generate all valid sequences for y
def valid_seq(y, targeted):
    x = []

    pairs = []
    unpairs = []
    stack = []
    for j, c in enumerate(y):
        if c == '(':
            stack.append(j)
        elif c == ')':
            pairs.append((stack.pop(), j))
        else:
            unpairs.append(j)

    if targeted:
        seq = ['A'] * len(y)
        for combination in product(['CG', 'GC'], repeat=len(pairs)):
            for idx, pair in enumerate(combination):
                i, j = pairs[idx]
                seq[i] = pair[0]
                seq[j] = pair[1]
            print("".join(seq))
    else:
        seq = ['A'] * len(y)
        for comb1 in product(['CG', 'GC', 'AU', 'UA', 'GU', 'UG'], repeat=len(pairs)):
            for comb2 in product(['A', 'C', 'G', 'U'], repeat=len(unpairs)):
                for idx, pair in enumerate(comb1):
                    i, j = pairs[idx]
                    seq[i] = pair[0]
                    seq[j] = pair[1]
                
                for idx, c in enumerate(comb2):
                    seq[unpairs[idx]] = c

                print("".join(seq))

valid_seq("((((...)))).", False)
# valid_seq("(((...)))", False)
# valid_seq("((...))((...))", False)

"""time cat /nfs/guille/huang/users/tangwe/seqs/n9_y.txt | ./LinearPartition/linearpartition -Vp 2>&1 | paste -d "\t" - - | cut -f5- -d' ' > /nfs/guille/huang/users/tangwe/Qx/n9_y.txt"""