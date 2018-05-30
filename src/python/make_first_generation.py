import sys
from scipy.stats import norm
import numpy as np
import random

def make_normal(n):
    ranges = []
    for i in range(1, 2**n + 1, 2):
        a,b = norm.interval(alpha=i/(2**n+2), loc=0, scale=1)
        ranges.append(a)
        ranges.append(b)
    ranges = np.asarray(ranges)
    ranges /= abs(max(ranges, key=abs))
    return np.sort(ranges) * random.uniform(0.1, 0.7)

def make_linear(n):
    return np.linspace(-1.0, 1.0, 2**n) * random.uniform(0.1, 1.0)

def make_log(n):
     ranges = np.concatenate((-1 * np.logspace(-1, 2.0, num=2**n),
                              np.logspace(-1, 2.0, num=2**n)))
     ranges = ranges[0::2]
     ranges /= abs(max(ranges, key=abs))
     return np.sort(ranges) * random.uniform(0.1, 0.7)

def make_random(n):
    ranges = np.concatenate((np.random.rand(2**(n-1)), np.random.rand(2**(n-1)) * -1))
    ranges /= abs(max(ranges, key=abs))
    return np.sort(ranges) * random.uniform(0.1, 0.7)

def main(bit, genom_num):
    genoms = [make_normal(bit), make_linear(bit), make_log(bit)]
    for _ in range(genom_num-3):
        genoms.append(make_random(bit))

    with open('src/ga/first_genoms.hpp', 'w') as f:
        f.write("#pragma once\n")
        f.write("#include <vector>\n\n")
        f.write("std::vector<std::vector<float>> range = {{")
        for i, genom in enumerate(genoms):
            if i != 0:
                f.write(", {")
            for j, v in enumerate(genom):
                if j != 0:
                    f.write(", ")
                f.write(str(v))
            f.write("}")
        f.write("};\n")

if __name__ =="__main__":
    main(4, 3)
