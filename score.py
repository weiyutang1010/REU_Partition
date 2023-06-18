def paired(c1, c2):
    if c1 + c2 in ["CG", "GC"]:
        return -1
    else:
        return -2

def unpaired(c):
    if c == 'A' or c == 'G':
        return .2
    else:
        return .1
