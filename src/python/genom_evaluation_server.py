from concurrent import futures
import sys
sys.path.append('../protos/')
import genom_pb2
import genom_pb2_grpc
import grpc

def calculate_fitness(genom):
    # 適応度を計算する処理
    fitness = 0.5
    return fitness

class GenomEvaluationServicer(genom_pb2_grpc.GenomEvaluationServicer):
    def GetIndividualWithEvaluation(self, request, context):
        return genom_pb2.Genom(genom=request,
                               evaluation=calculate_fitness(request))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor())
    genom_pb2_grpc.add_GenomEvaluationServicer_to_server(
        GenomEvaluationServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()

if __name__=='__main__':
    serve()
    
