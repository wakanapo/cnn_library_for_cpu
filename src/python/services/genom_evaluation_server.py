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

import cifar10
import imagenet
from quantize import VBQ

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
val_X = []
val_y = []

def converter(partition):
    def f(arr):
        arr = np.asarray(arr)
        for i in range(len(partition)-1):
            arr[(arr > partition[i]) & (arr <= partition[i+1])] = (partition[i] + partition[i + 1]) /  2
        arr[arr < partition[0]] = partition[0]
        arr[arr > partition[len(partition) - 1]] = partition[len(partition) - 1]
        return arr
    return f


def calculate_fitness(genom):
    K.clear_session()
    print("start evaluation!")
#     model = VGG16(weights='imagenet')
    model = cifar10.build_hinton_model(val_X.shape[1:])
    model.load_weights('data/hinton.h5')
    print("model load: success.")
#     W = model.get_weights()
#     W_q = list(map(converter(genom.gene), W))
#     print("quantize: success.")
#     model.set_weights(W_q)
#     model.compile(optimizer=optimizers.Adam(),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     predict = model.predict(val_X)
#     predict = np.argmax(predict, axis=1)
#     print("predict: ", predict[:5])
#     print("labels: ", val_y[:5])
#     print(val_y[:5]==predict[:5])
#     print("evaluate: success.")
#     return np.sum(val_y == predict) / len(val_y)
    score = model.evaluate(val_X, val_y)
    return score[1]

class GenomEvaluationServicer(genom_pb2_grpc.GenomEvaluationServicer):
    def GetIndividual(self, request, context):
        return genom_pb2.Individual(genom=request,
                               evaluation=calculate_fitness(request))

def serve():
    global val_X, val_y
#     val_X, val_y = imagenet.load()
    _, _, val_X, val_y = cifar10.read_data()
    print("data load: success.")
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

