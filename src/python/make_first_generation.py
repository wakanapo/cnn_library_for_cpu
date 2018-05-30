from scipy.stats import norm
import numpy as np
import random
import os
pwd = os.getcwd()
import sys
sys.path.append(pwd+'/src/protos')
import genom_pb2

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
    genes = [make_normal(bit), make_linear(bit), make_log(bit)]
    for _ in range(genom_num-3):
        genes.append(make_random(bit))

    message = genom_pb2.Genoms();
    for gene in genes:
        genoms = message.genoms.add()
        genoms.gene.extend(gene)

    with open(pwd+"/data/first_genom.pb", "wb") as f:
        f.write(message.SerializeToString())

if __name__ =="__main__":
    main(4, 3)
