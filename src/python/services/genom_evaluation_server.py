from concurrent import futures
import time
import os
pwd = os.getcwd()
import sys
sys.path.append(pwd+'/src/protos')
import genom_pb2
import genom_pb2_grpc
import grpc
from keras.applications.vgg16 import VGG16
import numpy as np

import imagenet
from quantize import VBQ

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
val_X = []
val_y = []

def converter(partition):
    def f(val):
        return VBQ(val, partition).to_float()
    return f

def calculate_fitness(genom):
    print("data load: success.")
    model = VGG16(include_top=True, weights='imagenet',
                  input_tensor=None, input_shape=None)
    print("model load: success.")
    W = model.get_weights()
    W_q = np.frompyfunc(converter(genom.gene), 1, 1)(W)
    print("quantize: success.")
    model.set_weights(W_q)
    score = model.evaluate(x=val_X, y=val_y)
    print("evaluate: success.")
    return score[1]

class GenomEvaluationServicer(genom_pb2_grpc.GenomEvaluationServicer):
    def GetIndividual(self, request, context):
        return genom_pb2.Individual(genom=request,
                               evaluation=calculate_fitness(request))

def serve():
    global val_X, val_y
    val_X, val_y = imagenet.load()
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
    serve()
    
