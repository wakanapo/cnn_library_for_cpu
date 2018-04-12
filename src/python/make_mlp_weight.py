import pickle
import numpy as np

with open('../../data/mnist_weight1_100.pickle', mode='rb') as f:
    w1 = pickle.load(f)

with open('../../data/mnist_weight2_100.pickle', mode='rb') as f:
    w2 = pickle.load(f)

with open('../../data/mnist_bias1_100.pickle', mode='rb') as f:
    b1 = pickle.load(f)

with open('../../data/mnist_bias2_100.pickle', mode='rb') as f:
    b2 = pickle.load(f)

with open('../mlp/mlp_weight.hpp', mode='w') as f:
    f.write("#pragma once\n")
    f.write("float w1_[] = ")
    f.write("{")
    for i, xs in enumerate(w1):
        if i != 0:
            f.write(", ")
        for j, x in enumerate(xs):
            if j != 0:
                f.write(", ")
            f.write(str(x))
    f.write("};\n")
    f.write("const int len_w1 = {};\n".format(len(w1)*len(w1[0])))
    f.write("const int dim_w1[] = {{{0}, {1}}};\n".format(len(w1[0]), len(w1)))
    f.write("float w2_[] = ")
    f.write("{")
    for i, xs in enumerate(w2):
        if i != 0:
            f.write(", ")
        for j, x in enumerate(xs):
            if j != 0:
                f.write(", ")
            f.write(str(x))
    f.write("};\n")
    f.write("const int len_w2 = {};\n".format(len(w2)*len(w2[0])))
    f.write("const int dim_w2[] = {{{0}, {1}}};\n".format(len(w2[0]), len(w2)))
    f.write("float b1_[] = ")
    f.write("{")
    for i, x in enumerate(b1):
        if i != 0:
            f.write(", ")
        f.write(str(x))
    f.write("};\n")
    f.write("const int len_b1 = {};\n".format(len(b1)))
    f.write("const int dim_b1[] = {{{0}, {1}}};\n".format(len(b1), 1))
    f.write("float b2_[] = ")
    f.write("{")
    for i, x in enumerate(b2):
        if i != 0:
            f.write(", ")
        f.write(str(x))
    f.write("};\n")
    f.write("const int len_b2 = {};\n".format(len(b2)))
    f.write("const int dim_b2[] = {{{0}, {1}}};\n".format(len(b2), 1))
