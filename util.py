import numpy as np
import math

class RNA:
    """Make Sequence or Distribution 1-indexed"""
    def __init__(self, seq):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return self.seq[idx-1]
    
    def __setitem__(self, idx, val):
        self.seq[idx-1] = val

    def __repr__(self):
        return self.seq

def print_dicts(text, array):
    # Print a list of dicts
    print(text)
    for i, x in enumerate(array):
        print(i, dict(sorted(x.items())))
    print()

# Generate catesian product of input, https://docs.python.org/3/library/itertools.html#itertools.product
def generate_sequences(*args, n):
    pools = [tuple(pool) for pool in args] * n
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

def generate_rand_distribution(n):
    test_distribution = []

    for _ in range(n):
        # generate random numbers that sum to 1: https://stackoverflow.com/a/8068956
        rand = np.concatenate((np.random.sample(3), np.array([1., 0.])), axis=0)
        # rand = np.array([round(x, 2) for x in rand])
        rand.sort()

        test_distribution.append({
            'A': rand[1] - rand[0],
            'C': rand[2] - rand[1],
            'G': rand[3] - rand[2],
            'U': rand[4] - rand[3]
        })

    return test_distribution

def generate_rand_structure(n):
    struct = []

    open_brackets = 0
    for i in range(n):
        if open_brackets >= n - i:
            struct.append(')')
            continue

        if i == n-1:
            struct.append('.')
            continue

        rand = 0
        if open_brackets == 0:
            rand = np.random.randint(0, 2)
        else:
            rand = np.random.randint(0, 3)

        if rand == 0:
            struct.append('.')
        elif rand == 1:
            struct.append('(')
            open_brackets += 1
        elif rand == 2:
            struct.append(')')
            open_brackets -= 1

    return "".join(struct)


def dicts_to_lists(X):
    lists = []
    for j, x in enumerate(X):
       lists.append(np.array([x['A'], x['C'], x['G'], x['U']]))
    return np.array(lists)

def probability(seq, X):
    prob = 1
    for idx, c in enumerate(seq, 1):
        prob *= X[idx][c]
    return prob

def verify_dicts(dict1, dict2):
    # dict1 and dict2 are list of defaultdicts
    n1, n2 = len(dict1), len(dict2)
    if n1 != n2:
        return False

    for idx in range(n1):
        for key in dict1[idx]:
            if not math.isclose(dict1[idx][key], dict2[idx][key]):
                return False
            
        for key in dict2[idx]:
            if not math.isclose(dict1[idx][key], dict2[idx][key]):
                return False
            
    return True

pairs = {'A': ['U'], 'G':['C', 'U'], 'U': ['A', 'G'], 'C': ['G']}
def generate_structure2(s, dp, num_brackets, left, right):
    s_length = right - left + 1
    if num_brackets == 0:
        return ["".join(['.' for _ in range(s_length)])]

    if s_length <= 1:
        return []

    res = []
    for length in range(s_length, 1, -1):
        for i in range(left, left + s_length - length + 1):
            j = i + length - 1
            
            if s[i] in pairs[s[j]]:
                if j == right and dp[i+1][j-1] + 1 >= num_brackets:
                    # generate structure between the brackets at i, j
                    new_st = generate_structure2(s, dp, num_brackets-1, i+1, j-1)
                    
                    # combine the results
                    for st in new_st:
                        temp = "".join(['.' for _ in range(i-left)])
                        temp += '(' + st + ')'
                        if temp not in res:
                            res.append(temp)

                if j < right and dp[i+1][j-1] + dp[j+1][right] + 1 >= num_brackets:
                    for x in range(num_brackets):
                        # generate structure between the brackets at i, j and after
                        new_st = generate_structure2(s, dp, x, i+1, j-1)
                        new_st2 = generate_structure2(s, dp, num_brackets - x - 1, j+1, right)

                        # combine the results
                        for st in new_st:
                            for st2 in new_st2:
                                temp = "".join(['.' for _ in range(i-left)])
                                temp += '(' + st + ')'
                                temp += st2
                                if temp not in res:
                                    res.append(temp)
                        
    return res

def all_structs(s):
    n = len(s)
    pq = []
    max_pairs_s = ['.' for i in range(n)]
    dp = [[0 for _ in range(n)] for _ in range(n)]

    # Perform dp O(n^3) time complexity, O(n^2) space complexity
    for k in range(1, n):
        for i in range(n - k):
            j = i + k

            dp[i][j] = dp[i][j-1]
            for t in range(i, j):
                if s[t] in pairs[s[j]]:
                    if t == 0:
                        dp[i][j] = max(dp[i][j], dp[t+1][j-1] + 1)
                    else:
                        dp[i][j] = max(dp[i][j], dp[i][t-1] + dp[t+1][j-1] + 1)

    # Generate k best structures using the table
    res = []
    for num_brac in range(dp[0][n-1], -1, -1):
        arr = generate_structure2(s, dp, num_brac, 0, n-1)
        for st in arr:
            res.append((num_brac, st))

    return res

# if __name__ == '__main__':
#     test = generate_rand_distribution(12)
#     for z in test:
#         print("{", end="")
#         for x, y in z.items():
#             print(f"{y:.2f}, ", end="")
#         print("},")
