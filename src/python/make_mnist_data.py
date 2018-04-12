from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
import numpy as np

mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data.astype('float32'),
                           mnist.target.astype('int32'), random_state=42)
mnist_X = mnist_X / 255.0
with open('../mlp/mnist_data.hpp', mode='w') as f:
    f.write("#pragma once\n")
    f.write("float x[{}] = ".format(100*len(mnist_X[0])))
    f.write("{")
    for i, xs in enumerate(mnist_X[:100]):
        if i != 0:
            f.write(", ")
        for j, x in enumerate(xs):
            if j != 0:
                f.write(", ")
            f.write(str(x))
    f.write("};\n")
    f.write("int image_num = {};\n".format(100))
    f.write("int image_size = {};\n".format(len(mnist_X[0])))
    f.write("unsigned long t[{}] = ".format(100))
    f.write("{")
    for i, x in enumerate(mnist_y[:100]):
        if i != 0:
            f.write(", ")
        f.write(str(x))
    f.write("};\n")
