import os
home=os.environ['HOME']
import sys
sys.path.append(home+'/utokyo-kudohlab/cnn_cpp/src/protos')
import gzip
import arithmatic_pb2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

calcs = arithmatic_pb2.One()

def SaveGraph(df, f, name, i):
    _, (axL, axC, axR) = plt.subplots(ncols=3, figsize=(40, 10), sharex=True, sharey=True)
    axL.hist(f(df.a), bins=min(len(df[:]), 500), color='deepskyblue')
    axL.set_title('a')
    axL.savefig(name+"_a_{}.png".format(i))
    axC.hist(f(df.b), bins=min(len(df[:]), 500), color='hotpink')
    axC.set_title('b')
    axL.savefig(name+"_b_{}.png".format(i))
    axR.hist(f(df.ans), bins=min(len(df[:]), 500), color='orange')
    axR.set_title('ans')
    axL.savefig(name+"_ans_{}.png".format(i))

def SaveCSV(df, name, i):
    df.to_csv(name+'_{}.csv'.format(i), index=False)

def f(x):
    return x

def log_10(x):
    return np.log10(x[np.nonzero(x)]).dropna()


for i in range(15000):
    try:
        with gzip.open(home+"/utokyo-kudohlab/cnn_cpp/data/10E_1/10E_1_{}.pb".format(i), "rb") as f:
            calcs.ParseFromString(f.read())
    except IOError:
        print ("Could not open file.  Creating a new one.")

    df = pd.DataFrame([(x.file[x.file.find("src/")+4:]+str(x.line), x.operator, x.a, x.b, x.ans) for x in calcs.calc])
    df.columns = ["position", "operator", "a", "b", "ans"]
    df_f = pd.read_csv(home+'/utokyo-kudohlab/cnn_cpp/data/cnn_function.csv')
    df_f = df_f.assign(position=df_f.file+df_f.line.apply(str))
    df = pd.merge(df, df_f, on='position', how='left')
    df_add = df[df.operator=="+"]
    df_sub = df[df.operator=="-"]
    df_mul = df[df.operator=="*"]
    df_div = df[df.operator=="/"]

    name = "all_value"
    SaveGraph(df, f, name, i)
    SaveCSV(df.loc[:, ['a', 'b', 'ans']].describe(), name, i)

    name = "all_value_log10"
    SaveGraph(df, log_10, name, i)
    SaveCSV(df.loc[:, ['a', 'b', 'ans']].describe(), name, i)

    df_add_n = df_add[df_add.a*df_add.b < 0]
    df_add_p = df_add[df_add.a*df_add.b > 0]
    df_sub_n = df_sub[df_sub.a*df_sub.b < 0]
    df_sub_p = df_sub[df_sub.a*df_sub.b > 0]

    _, (axL, axR) = plt.subplots(ncols=2, figsize=(40, 10))
    axL.hist(np.log10((df_add_n.a*10000 + df_add_n.b*10000).append(df_sub_p.a*10000 - df_sub_p.b*10000)).dropna(), bins=500, color='lightgreen')
    axL.savefig("cancellation.png")
    axR.hist((abs(np.log10(df_add_p.a) - np.log10(df_add_p.b)).append(abs(np.log10(df_sub_n.a) - np.log10(df_sub_n.b)))).dropna(), bins=500, color='salmon')
    axR.savefig("quantization.png")

    block = [0, 864000, 881280, 1745180, 1745280, 1747270, 1747280, 1747320, 1747330, 1750230,
         1751230, 1752230, 1752240, 1752250, 1752260, 1752360, 3044040, 3476040, 3908040,
         3908140, 3908240, 3908340, 3925620, 5101620, 5965620, 5966370, 5967120, 5984400,
         5984430]

    start = block[0]

    funcs = []
    for end in block[1:]:
        funcs.append((df.function[start], df[start:end]))
        start = end
    funcs.append((df.function[start], df[start:]))

    for k, v in funcs:
        SaveGraph(v, log_10, k, i)
        SaveCSV(df.loc[:, ['a', 'b', 'ans']].describe(), name, i)
