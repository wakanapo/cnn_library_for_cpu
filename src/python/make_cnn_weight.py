import numpy as np
import pickle

with open("../../data/params.pkl", 'rb') as fr:
    params = pickle.load(fr)
    with open('../cnn/cnn_weight.hpp', mode='w') as fw:
        fw.write("#pragma once\n\n")
        fw.write("float w1_raw[] = {")
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
        fw.write("const int dim_w1[] = {{{0}, {1}, {2}, {3}}};\n".format(len(params['W1'][0][0][0]), len(params['W1'][0][0]), len(params['W1'][0]), len(params['W1'])))
        fw.write("const int len_w1 = {};\n".format(len(params['W1'][0][0][0])*len(params['W1'][0][0])*len(params['W1'][0])*len(params['W1'])))
        fw.write("float b1_raw[] = ")
        fw.write("{")
        for i, x in enumerate(params['b1']):
            if i != 0:
                fw.write(", ")
            fw.write(str(x))
        fw.write("};\n")
        fw.write("const int len_b1 = {};\n".format(len(params['b1'])))
        fw.write("const int dim_b1[] = {{{}}};\n".format(len(params['b1'])))
        fw.write("float b2_raw[] = ")
        fw.write("{")
        for i, x in enumerate(params['b2']):
            if i != 0:
                fw.write(", ")
            fw.write(str(x))
        fw.write("};\n")
        fw.write("const int len_b2 = {};\n".format(len(params['b2'])))
        fw.write("const int dim_b2[] = {{{0}, {1}}};\n".format(len(params['b2']), 1))
        fw.write("float b3_raw[] = ")
        fw.write("{")
        for i, x in enumerate(params['b3']):
            if i != 0:
                fw.write(", ")
            fw.write(str(x))
        fw.write("};\n")
        fw.write("const int len_b3 = {};\n".format(len(params['b3'])))
        fw.write("const int dim_b3[] = {{{0}, {1}}};\n".format(len(params['b3']), 1))
        fw.write("float w2_raw[] = ")
        fw.write("{")
        for i, xs in enumerate(params['W2']):
            if i != 0:
                fw.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    fw.write(", ")
                fw.write(str(x))
        fw.write("};\n")
        fw.write("const int len_w2 = {};\n".format(len(params['W2'])*len(params['W2'][0])))
        fw.write("const int dim_w2[] = {{{0}, {1}}};\n".format(len(params['W2'][0]), len(params['W2'])))
        fw.write("float w3_raw[] = ")
        fw.write("{")
        for i, xs in enumerate(params['W3']):
            if i != 0:
                fw.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    fw.write(", ")
                fw.write(str(x))
        fw.write("};\n")
        fw.write("const int len_w3 = {};\n".format(len(params['W3'])*len(params['W3'][0])))
        fw.write("const int dim_w3[] = {{{0}, {1}}};\n".format(len(params['W3'][0]), len(params['W3'])))        

### Using Tensorflow weight sample is below.
        
# import numpy as np
# import tensorflow as tf

# w1 = tf.get_variable("w1", shape=(5, 5, 1, 20))
# b1 = tf.get_variable("b1", shape=[20])
# w2 = tf.get_variable("w2", shape=(5, 5, 20, 50))
# b2 = tf.get_variable("b2", shape=[50])
# fc_w = tf.get_variable("fc_w", shape=(4*4*50, 10))
# fc_b = tf.get_variable("fc_b", shape=[10])
# saver = tf.train.Saver()

# with tf.Session() as sess:
#     saver.restore(sess, "/home/wakana/AIL/chap7/model.ckpt")
#     with open('../cpp/cnn_weight.hpp', mode='w') as f:
#         f.write("#pragma once\n\n")
#         f.write("float w1_raw[] = {")
#         for out_ch in range(len(w1.eval()[0][0][0])):
#             for in_ch in range(len(w1.eval()[0][0])):
#                 for j in range(len(w1.eval()[0])):  # 列数
#                     for i in range(len(w1.eval())):  # 行数
#                         if (i+j+in_ch+out_ch) != 0:
#                             f.write(", ")
#                         x = w1.eval()[i, j, in_ch, out_ch]
#                         f.write(str(x))
#         f.write("};\n")
#         f.write("const int dim_w1[] = {{{0}, {1}, {2}, {3}}};\n".format(
#             len(w1.eval().T[0][0][0]), len(w1.eval().T[0][0]),
#             len(w1.eval().T[0]), len(w1.eval().T)))
#         f.write("const int len_w1 = {};\n".format(len(w1.eval().T[0][0][0])*len(w1.eval().T[0][0])*
#                                                   len(w1.eval().T[0])*len(w1.eval().T)))
#         f.write("float w2_raw[] = {")
#         for out_ch in range(len(w2.eval()[0][0][0])):
#             for in_ch in range(len(w2.eval()[0][0])):
#                 for j in range(len(w2.eval()[0])):  # 列数
#                     for i in range(len(w2.eval())):  # 行数
#                         if (i+j+in_ch+out_ch) != 0:
#                             f.write(", ")
#                         x = w2.eval()[i, j, in_ch, out_ch]
#                         f.write(str(x))
#         f.write("};\n")
#         f.write("const int dim_w2[] = {{{0}, {1}, {2}, {3}}};\n".format(
#             len(w2.eval().T[0][0][0]), len(w2.eval().T[0][0]),
#             len(w2.eval().T[0]), len(w2.eval().T)))
#         f.write("const int len_w2 = {};\n".format(len(w2.eval().T[0][0][0])*len(w2.eval().T[0][0])*
#                                                   len(w2.eval().T[0])*len(w2.eval().T)))
#         f.write("float b1_raw[] = ")
#         f.write("{")
#         for i, x in enumerate(b1.eval()):
#             if i != 0:
#                 f.write(", ")
#             f.write(str(x))
#         f.write("};\n")
#         f.write("float b2_raw[] = ")
#         f.write("{")
#         for i, x in enumerate(b2.eval()):
#             if i != 0:
#                 f.write(", ")
#             f.write(str(x))
#         f.write("};\n")
#         new_fc_w = tf.reshape(fc_w, [4, 4, 50, 10])
#         f.write("float fc_w_raw[] = ")
#         f.write("{")
#         for i in range(50):
#             for j in range(4):
#                 for k in range(4):
#                     if (i+j+k) != 0:
#                         f.write(", ")
#                     xs = new_fc_w.eval()[k,j,i]
#                     for k, x in enumerate(xs):
#                         if k != 0:
#                             f.write(", ")
#                         f.write(str(x))
#         f.write("};\n")
#         f.write("const int dim_fc_w[] = {{{0}, {1}}};\n".format(len(fc_w.eval()[0]), (len(fc_w.eval()))))
#         f.write("const int len_fc_w = {};\n".format(len(fc_w.eval())*(len(fc_w.eval()[0]))))
#         f.write("float fc_b_raw[] = ")
#         f.write("{")
#         for i, x in enumerate(fc_b.eval()):
#             if i != 0:
#                 f.write(", ")
#             f.write(str(x))
#         f.write("};\n")
#         f.write("const int dim_fc_b[] = {{{}}};\n".format(len(fc_b.eval()), 1))
#         f.write("const int len_fc_b = {};\n".format(len(fc_b.eval())))
