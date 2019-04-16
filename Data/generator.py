# Generate custom datasets, that are easy to interpret the results of and easy to train on.

# From AI-programming course code (Keth Downing)
# Produce a list of pairs, with each pair consisting of a num_bits bit pattern and a singleton list containing
# the parity bit: 0 => an even number of 1's, 1 => odd number of 1's.  When double=True, a 2-bit vector is the
# target, with bit 0 indicating even parity and bit 1 indicating odd parity.
import numpy as np
def gen_all_parity_cases(num_bits=10, double=True): # 10 bits, that can be double.
    def parity(v): return sum(v) % 2
    def target(v):
        if double:
            tg = [0,0].copy()
            tg[parity(v)] = 1
            return tg
        else: return [parity(v)]
    return [[c, target(c)] for c in gen_all_bit_vectors(num_bits)] # return a list of lists.

# Generate all bit vectors of a given length (num_bits).
def gen_all_bit_vectors(num_bits):
    def bits(n):
        s = bin(n)[2:]
        return [int(b) for b in '0'*(num_bits - len(s))+s]
    return [bits(i) for i in range(2**num_bits)]

def gen_custom_function_dataset(num_features, num_labels): #
    # Simple function, that generate a dataset depending on the number of features we want.
    # f(X) = Y , X = {x_1,x_2,...,x_i}, Y = {y_1,y_2,...,y_j}   
    # This cannot be random
    # We want to create a function that that given x, return y.
    f = lambda x,y : x+y

    return f(num_features,num_labels)