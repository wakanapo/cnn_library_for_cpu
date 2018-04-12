import os
home = os.environ['HOME']
import sys
sys.path.append(home+'/utokyo-kudohlab/cnn_cpp/src/protos')
import genom_pb2
import numpy as np
import matplotlib.pyplot as plt

def main(filename, n):
    genoms = genom_pb2.Genoms()
    colors = np.random.rand(100, 3, 1)
    for j in range(n):
        try:
            with open(home+"/utokyo-kudohlab/cnn_cpp/data/{0}{1}.pb"
                      .format(filename, j),
                      "rb") as f:
                genoms.ParseFromString(f.read())
        except IOError:
            print ("Could not open file.")
            
        for i in range(len(genoms.genoms)):
            plt.scatter(genoms.genoms[i].gene,
                        np.full_like(genoms.genoms[i].gene, genoms.genoms[i].evaluation),
                        c=colors[i])

        plt.xlim(-1.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.title("genoms")
        plt.ylabel("accuracy")
        plt.xlabel("range")
        plt.savefig(home+"/utokyo-kudohlab/cnn_cpp/data/evaluation_{0}{1}.png"
                    .format(filename, j))
        plt.close()

if __name__=="__main__":
    argv = sys.argv
    if len(argv) != 2:
        print("Usage: Python {} filename".format(argv[0]))
        quit()
    main(argv[1], 50)
