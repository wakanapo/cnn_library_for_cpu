import numpy as np
import pickle

with open("../../data/before.pkl", 'rb') as fr:
    params = pickle.load(fr)
    with open('../../test/cnn_init_weight.hpp', mode='w') as fw:
        fw.write("#pragma once\n\n")
        fw.write("float iw1_raw[] = {")
        for i, out_ch in enumerate(params['W1']):
            if i != 0:
                fw.write(", ")
            for j, in_ch in enumerate(out_ch):
                if j != 0:
                    fw.write(", ")
                for k, xs in enumerate(in_ch):
                    if k != 0:
                        fw.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            fw.write(", ")
                        fw.write(str(x))
        fw.write("};\n")
        fw.write("const int dim_iw1[] = {{{0}, {1}, {2}, {3}}};\n".format(len(params['W1'][0][0][0]), len(params['W1'][0][0]), len(params['W1'][0]), len(params['W1'])))
        fw.write("const int len_iw1 = {};\n".format(len(params['W1'][0][0][0])*len(params['W1'][0][0])*len(params['W1'][0])*len(params['W1'])))
        fw.write("float ib1_raw[] = ")
        fw.write("{")
        for i, x in enumerate(params['b1']):
            if i != 0:
                fw.write(", ")
            fw.write(str(x))
        fw.write("};\n")
        fw.write("const int len_ib1 = {};\n".format(len(params['b1'])))
        fw.write("const int dim_ib1[] = {{{}}};\n".format(len(params['b1'])))
        fw.write("float ib2_raw[] = ")
        fw.write("{")
        for i, x in enumerate(params['b2']):
            if i != 0:
                fw.write(", ")
            fw.write(str(x))
        fw.write("};\n")
        fw.write("const int len_ib2 = {};\n".format(len(params['b2'])))
        fw.write("const int dim_ib2[] = {{{0}, {1}}};\n".format(len(params['b2']), 1))
        fw.write("float ib3_raw[] = ")
        fw.write("{")
        for i, x in enumerate(params['b3']):
            if i != 0:
                fw.write(", ")
            fw.write(str(x))
        fw.write("};\n")
        fw.write("const int len_ib3 = {};\n".format(len(params['b3'])))
        fw.write("const int dim_ib3[] = {{{0}, {1}}};\n".format(len(params['b3']), 1))
        fw.write("float iw2_raw[] = ")
        fw.write("{")
        for i, xs in enumerate(params['W2']):
            if i != 0:
                fw.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    fw.write(", ")
                fw.write(str(x))
        fw.write("};\n")
        fw.write("const int len_iw2 = {};\n".format(len(params['W2'])*len(params['W2'][0])))
        fw.write("const int dim_iw2[] = {{{0}, {1}}};\n".format(len(params['W2'][0]), len(params['W2'])))
        fw.write("float iw3_raw[] = ")
        fw.write("{")
        for i, xs in enumerate(params['W3']):
            if i != 0:
                fw.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    fw.write(", ")
                fw.write(str(x))
        fw.write("};\n")
        fw.write("const int len_iw3 = {};\n".format(len(params['W3'])*len(params['W3'][0])))
        fw.write("const int dim_iw3[] = {{{0}, {1}}};\n".format(len(params['W3'][0]), len(params['W3'])))        
