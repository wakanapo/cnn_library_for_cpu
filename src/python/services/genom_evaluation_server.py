import copy
from concurrent import futures
import time
import os
pwd = os.getcwd()
import sys
sys.path.append(pwd+'/src/protos')
import genom_pb2
import genom_pb2_grpc
import grpc
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import backend as K
from keras import optimizers
import numpy as np
import tensorflow as tf

import cifar10
import imagenet
from quantize import VBQ

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
val_X = []
val_y = []
g_W = []

def converter(partition):
    def f(arr):
        arr = np.asarray(arr)
        end_idx = len(partition) - 1
        for i in range(end_idx):
            arr[(arr > partition[i]) & (arr <= partition[i+1])] = (partition[i] + partition[i + 1]) /  2
        arr[arr <= partition[0]] = partition[0]
        arr[arr > partition[end_idx]] = partition[end_idx]
        return arr
    return f


def calculate_fitness(genom):
    with K.get_session().graph.as_default():
        print("start evaluation!")
        #     model = VGG16(weights='imagenet')
        model = cifar10.Vgg_like().build(val_X.shape[1:])
        W_q = list(map(converter(genom.gene), copy.deepcopy(g_W)))
        print("quantize: success.")
        model.set_weights(W_q)
        model.compile(optimizer=optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        score = model.evaluate(val_X[g_offset:g_offset+5000], val_y[g_offset:g_offset+5000])
    K.clear_session()
    return score[1]

class GenomEvaluationServicer(genom_pb2_grpc.GenomEvaluationServicer):
    def GetIndividual(self, request, context):
        return genom_pb2.Individual(genom=request,
                               evaluation=calculate_fitness(request))

def serve():
    global val_X, val_y, g_W
#     val_X, val_y = imagenet.load()
    _, _, val_X, val_y = cifar10.read_data()
    print("data load: success.")
    model_class = cifar10.Vgg_like()
    model = model_class.build(val_X.shape[1:])
    model.load_weights('data/'+model_class.name+'.h5')
    g_W = model.get_weights()
    print("model load: success.")
    server = grpc.server(futures.ThreadPoolExecutor())
    genom_pb2_grpc.add_GenomEvaluationServicer_to_server(
        GenomEvaluationServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__=='__main__':
    argv = sys.argv
    if len(argv) > 1:
        g_offset = int(sys.argv[1])
    else:
        g_offset = 0
    serve(offset)

