from scipy.stats import norm
import numpy as np

def make_vec(n):
    a, b = norm.interval(alpha=(2**n)/(2**n+1), loc=0, scale=0.4)
    return np.linspace(a, b, 2**(n+1))

def main():
    ranges = make_vec(1)
    with open('../util/box_linear.hpp', 'w') as f:
        f.write("#pragma once\n")
        f.write("#include <vector>\n\n")
        f.write("std::vector<float> range = {")
        for i, v in enumerate(ranges):
            if i != 0:
                f.write(", ")
            f.write(str(v))
        f.write("};\n")

if __name__=="__main__":
    main()
