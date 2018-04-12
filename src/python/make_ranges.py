from scipy.stats import norm

def make_vec(n):
    ranges = []
    for i in range(1, 2**n + 1):
        a,b = norm.interval(alpha=i/(2**n+1), loc=0, scale=0.4)
        ranges.append(a)
        ranges.append(b)
    return sorted(ranges)

def main():
    ranges = make_vec(7)
    with open('../util/box_range.hpp', 'w') as f:
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
