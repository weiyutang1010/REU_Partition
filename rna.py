from heapq import *

best_test_cases = ["ACAGU", "GCACG", "UUCAGGA", "GUUAGAGUCU", "AUAACCUUAUAGGGCUCUG",
                   "AACCGCUGUGUCAAGCCCAUCCUGCCUUGUU", "GAUGCCGUGUAGUCCAAAGACUUCACCGUUGG",
                   "CAUCGGGGUCUGAGAUGGCCAUGAAGGGCACGUACUGUUU",
                   "ACGGCCAGUAAAGGUCAUAUACGCGGAAUGACAGGUCUAUCUAC",
                   "AGGCAUCAAACCCUGCAUGGGAGCACCGCCACUGGCGAUUUUGGUA"]

test_cases = ["ACAGU", "AC", "GUAC", "GCACG", "CCGG", "CCCGGG", "UUCAGGA",
              "AUAACCUA", "UUGGACUUG", "UUUGGCACUA", "GAUGCCGUGUAGUCCAAAGACUUC",
              "AGGCAUCAAACCCUGCAUGGGAGCG"]
pairs = {'A': ['U'], 'G':['C', 'U'], 'U': ['A', 'G'], 'C': ['G']}
    

# Generate the best structure recursively using the dp table
def generate_structure(s, max_pairs_s, dp, left, right, num_brackets):
    if num_brackets == 0:
        return

    for length in range(right - left + 1, 1, -1):
        for i in range(left, right - length + 2):
            j = i + length - 1
            
            if s[i] in pairs[s[j]]:
                if j == right and dp[i+1][j-1] + 1 == num_brackets:
                    # generate structure between the brackets
                    max_pairs_s[i] = '('
                    max_pairs_s[j] = ')'

                    generate_structure(s, max_pairs_s, dp, i+1, j-1, dp[i+1][j-1])
                    return

                if j < right and dp[i+1][j-1] + dp[j+1][right] + 1 == num_brackets:
                    # generate structure between the brackets and after
                    max_pairs_s[i] = '('
                    max_pairs_s[j] = ')'

                    generate_structure(s, max_pairs_s, dp, i+1, j-1,  dp[i+1][j-1])
                    generate_structure(s, max_pairs_s, dp, j+1, right, dp[j+1][right])
                    return    


def best(s):
    max_pairs_s = ['.' for i in range(len(s))]

    dp = [[0 for _ in range(len(s))] for _ in range(len(s))]

    # Perform dp O(n^3) time complexity, O(n^2) space complexity
    for k in range(1, len(s)):
        for i in range(len(s) - k):
            j = i + k

            dp[i][j] = dp[i][j-1]
            for t in range(i, j):
                if s[t] in pairs[s[j]]:
                    if t == 0:
                        dp[i][j] = max(dp[i][j], dp[t+1][j-1] + 1)
                    else:
                        dp[i][j] = max(dp[i][j], dp[i][t-1] + dp[t+1][j-1] + 1)

    # Recursively generate brackets, time complexity should be O(n^2)
    generate_structure(s, max_pairs_s, dp, 0, len(s)-1, dp[0][len(s)-1])
    return (dp[0][len(s)-1], "".join(max_pairs_s))

def total(s):
    num_structures = 0

    dp = [[1 for _ in range(len(s))] for _ in range(len(s))]

    # Perform dp O(n^3) time complexity, O(n^2) space complexity
    for k in range(1, len(s)):
        for i in range(len(s) - k):
            j = i + k

            dp[i][j] = dp[i][j-1]
            for t in range(i, j):
                if s[t] in pairs[s[j]]:
                    if t == 0:
                        dp[i][j] += dp[t+1][j-1]
                    else:
                        dp[i][j] += dp[i][t-1] * dp[t+1][j-1]

    return dp[0][len(s)-1]

# Generate k structures with n brackets and return result as a list of strings
def generate_structure2(s, dp, num_brackets, left, right, k):
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
                    new_st = generate_structure2(s, dp, num_brackets-1, i+1, j-1, k)
                    
                    # combine the results
                    for st in new_st:
                        temp = "".join(['.' for _ in range(i-left)])
                        temp += '(' + st + ')'
                        if temp not in res:
                            res.append(temp)

                        if len(res) >= k:
                            # terminate early if there is enough result
                            return res

                if j < right and dp[i+1][j-1] + dp[j+1][right] + 1 >= num_brackets:
                    for x in range(num_brackets):
                        # generate structure between the brackets at i, j and after
                        new_st = generate_structure2(s, dp, x, i+1, j-1, k)
                        new_st2 = generate_structure2(s, dp, num_brackets - x - 1, j+1, right, k)

                        # combine the results
                        for st in new_st:
                            for st2 in new_st2:
                                temp = "".join(['.' for _ in range(i-left)])
                                temp += '(' + st + ')'
                                temp += st2
                                if temp not in res:
                                    res.append(temp)

                                if len(res) >= k:
                                    # terminate early if there is enough result
                                    return res
                        
    return res


def kbest(s, n):
    pq = []
    max_pairs_s = ['.' for i in range(len(s))]
    dp = [[0 for _ in range(len(s))] for _ in range(len(s))]

    # Perform dp O(n^3) time complexity, O(n^2) space complexity
    for k in range(1, len(s)):
        for i in range(len(s) - k):
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
    for i in range(dp[0][len(s)-1], -1, -1):
        arr = generate_structure2(s, dp, i, 0, len(s)-1, n)
        for st in arr:
            res.append((i, st))

            if len(res) == n:
                return res

    return res

def run_best_test():
    for test in best_test_cases:
        print(test)
        print(best(test))

def run_all_tests():
    for test in test_cases:
        print(test)
        print(best(test))
        print(total(test))
        print(kbest(test, 10))
        print("------")

if __name__ == "__main__":
    run_best_test()
    run_all_tests()
    