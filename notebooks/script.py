from advas.phonetics import Phonetics
import numpy as np


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


# -- initialize phonetics object


# -- weight
weight = {
    "soundex": 0.2,
    "caverphone": 0.2,
    "metaphone": 0.5,
    "nysiis": 0.1
}

# -- algorithms
algorithms = ["soundex", "caverphone", "metaphone", "nysiis"]


def get_similarity(word1, word2):
    word1 = Phonetics(word1)
    word2 = Phonetics(word2)

    codeList1 = word1.phoneticCode()
    codeList2 = word2.phoneticCode()
    total = 0.0
    for entry in algorithms:
        code1 = codeList1[entry]
        code2 = codeList2[entry]
        lev = levenshtein(code1, code2)
        currentWeight = weight[entry]
        # print("comparing %s with %s for %s (%0.2f: weight %0.2f)" % (code1, code2, entry, lev, currentWeight))
        subtotal = lev * currentWeight
        total += subtotal

    return total
