import numpy as np
import pickle

with open("../../data/before.pkl", 'rb') as fr:
    params = pickle.load(fr)
    with open("../../data/grad.pkl", 'rb') as fg:
        grads = pickle.load(fg)
        with open('../../test/cnn_new_weight.hpp', mode='w') as fw:
            fw.write("#pragma once\n\n")
            fw.write("float nw1_raw[] = {")
            for i, out_ch in enumerate(params['W1']-grads['W1']):
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
            fw.write("const int dim_nw1[] = {{{0}, {1}, {2}, {3}}};\n".format(len(params['W1'][0][0][0]), len(params['W1'][0][0]), len(params['W1'][0]), len(params['W1'])))
            fw.write("const int len_nw1 = {};\n".format(len(params['W1'][0][0][0])*len(params['W1'][0][0])*len(params['W1'][0])*len(params['W1'])))
            fw.write("float nb1_raw[] = ")
            fw.write("{")
            for i, x in enumerate(params['b1']-grads['b1']):
                if i != 0:
                    fw.write(", ")
                fw.write(str(x))
            fw.write("};\n")
            fw.write("const int len_nb1 = {};\n".format(len(params['b1'])))
            fw.write("const int dim_nb1[] = {{{}}};\n".format(len(params['b1'])))
            fw.write("float nb2_raw[] = ")
            fw.write("{")
            for i, x in enumerate(params['b2']-grads['b2']):
                if i != 0:
                    fw.write(", ")
                fw.write(str(x))
            fw.write("};\n")
            fw.write("const int len_nb2 = {};\n".format(len(params['b2'])))
            fw.write("const int dim_nb2[] = {{{0}, {1}}};\n".format(len(params['b2']), 1))
            fw.write("float nb3_raw[] = ")
            fw.write("{")
            for i, x in enumerate(params['b3']-grads['b3']):
                if i != 0:
                    fw.write(", ")
                fw.write(str(x))
            fw.write("};\n")
            fw.write("const int len_nb3 = {};\n".format(len(params['b3'])))
            fw.write("const int dim_nb3[] = {{{0}, {1}}};\n".format(len(params['b3']), 1))
            fw.write("float nw2_raw[] = ")
            fw.write("{")
            for i, xs in enumerate(params['W2']-grads['W2']):
                if i != 0:
                    fw.write(", ")
                for j, x in enumerate(xs):
                    if j != 0:
                        fw.write(", ")
                    fw.write(str(x))
            fw.write("};\n")
            fw.write("const int len_nw2 = {};\n".format(len(params['W2'])*len(params['W2'][0])))
            fw.write("const int dim_nw2[] = {{{0}, {1}}};\n".format(len(params['W2'][0]), len(params['W2'])))
            fw.write("float nw3_raw[] = ")
            fw.write("{")
            for i, xs in enumerate(params['W3']-grads['W3']):
                if i != 0:
                    fw.write(", ")
                for j, x in enumerate(xs):
                    if j != 0:
                        fw.write(", ")
                    fw.write(str(x))
            fw.write("};\n")
            fw.write("const int len_nw3 = {};\n".format(len(params['W3'])*len(params['W3'][0])))
            fw.write("const int dim_nw3[] = {{{0}, {1}}};\n".format(len(params['W3'][0]), len(params['W3'])))        
        
