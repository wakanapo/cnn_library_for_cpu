import os
pwd = os.getcwd()
import sys
sys.path.append(pwd+'/src/protos')
import genom_pb2
import numpy as np
import matplotlib.pyplot as plt

def main(filename, n):
    genoms = genom_pb2.Genoms()
    colors = np.random.rand(100, 3)
    for j in range(n):
        try:
            with open(pwd+"/data/{0}/{0}{1}.pb".format(filename, j), "rb") as f:
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
        plt.savefig(pwd+"/data/{0}/evaluation_{1}.png".format(filename, j))
        plt.close()
        
        for i in range(len(genoms.genoms)):
            plt.scatter(genoms.genoms[i].gene,
                        np.full_like(genoms.genoms[i].gene, i),
                        c=colors[i])

        plt.xlim(-1.0, 1.0)
        plt.ylim(-1, n+1)
        plt.title("genoms")
        plt.ylabel("genoms #")
        plt.xlabel("range")
        plt.savefig(pwd+"/data/{0}/genoms_{1}.png".format(filename, j))
        plt.close()

if __name__=="__main__":
    argv = sys.argv
    if len(argv) != 3:
        print("Usage: Python {} filename #genom".format(argv[0]))
        quit()
    main(argv[1], int(argv[2]))

