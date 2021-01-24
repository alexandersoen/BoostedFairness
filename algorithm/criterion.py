import math

def entropy(q):
    return -q * math.log2(q) - (1 - q) * math.log2(1 - q)

def gini(q):
    return 4 * q * (1 - q)