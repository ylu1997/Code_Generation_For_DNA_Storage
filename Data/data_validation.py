# data_validation.py

"""
This Python file contains functions for checking data correctness and computing distances between sequences.
The four main functions included are:
1. lcs_distance(s1, s2): Calculate the Longest Common Subsequence (LCS) distance between two sequences.
2. hamming_distance(s1, s2): Calculate the Hamming distance between two sequences.
3. levenshtein_distance(s1, s2): Calculate the Levenshtein distance between two sequences.
4. mini_distance(codebook, distance_function): Calculate the minimum distance between all pairs of different sequences in the codebook.

These functions can be used for various data validation and distance computation tasks.

Note: If the codebook is intended to correct n occurrences of a specific type of error,
the mini_distance function should be at least greater than 2n to ensure effective error correction.
"""

def lcs_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    lcs_distance = m + n - 2 * lcs_length

    return lcs_distance

def hamming_distance(s1, s2):
    if len(str1) != len(str2):
        raise ValueError("Input strings must have the same length")

    distance = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    return distance
  
def levenshtein_distance(s1,s2):
    len_s1 = len(s1)
    len_s2 = len(s2)
 
    matrix = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
 
    for i in range(len_s1 + 1):
        matrix[i][0] = i
    for j in range(len_s2 + 1):
        matrix[0][j] = j
 
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,  # deletion
                matrix[i][j - 1] + 1,  # insertion
                matrix[i - 1][j - 1] + cost  # substitution
            )
 
    return matrix[len_s1][len_s2]

def mini_distance(codebook,distance_function):
    result = float('inf' )
    N = len(codebook)
    for i in range(N):
        s1 = codebook[i]
        for j in range(i+1,N):
            s2 = codebook[j]
            d = distance_function(s1,s2)
            print(d,(i,j))
            if d<result:
                result = d
    return result
