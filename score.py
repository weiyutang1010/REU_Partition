def paired(c1, c2):
    _allowed_pairs = {"CG": -3, "GC": -3, "AU": -2, "UA":-2, "GU": -1, "UG":-1}
    if c1 + c2 in _allowed_pairs:
        return _allowed_pairs[c1 + c2]
    else:
        return 1000

def unpaired(c):
    return 1.
