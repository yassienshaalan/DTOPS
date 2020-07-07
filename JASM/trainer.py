from neuralnet import *
from fastdropoutnet import *
from dbm import *
from dbn import *
from sparse_coder import *
from choose_matrix_library import *
import numpy as np
from time import sleep
from cudamat import cudamat as cm
cm.cublas_init()
def LockGPU(max_retries=10):
  for retry_count in range(max_retries):
    board = gpu_lock.obtain_lock_id()
    board = 0
    #print("board "+str(board))
    if board != -1:
      break
    sleep(1)
  
  if board == -1:
    print('No GPU board available.')
    #sys.exit(1)
  else:
    #print("Gpu supported")
    cm.cuda_set_device(board)
    cm.cublas_init()
  return board

def FreeGPU(board):
  cm.cublas_shutdown()
  #gpu_lock.free_lock(board)

def LoadExperiment(model_file, train_op_file, eval_op_file):
  model = util.ReadModel(model_file)
  train_op = util.ReadOperation(train_op_file)
  eval_op = util.ReadOperation(eval_op_file)
  return model, train_op, eval_op

def CreateDeepnet(model, train_op, eval_op):
  
  if model.model_type == deepnet_pb2.Model.FEED_FORWARD_NET:
    print("Creating FEED_FORWARD_NET")
    return NeuralNet(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.DBM:
    print("Creating DBM",model.model_type)
    return DBM(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.DBN:
    print("Creating DBN")
    return DBN(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.SPARSE_CODER:
    print("Creating SPARSE_CODER")
    return SparseCoder(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.FAST_DROPOUT_NET:
    print("Creating FAST_DROPOUT_NET")
    return FastDropoutNet(model, train_op, eval_op)
  else:
    print("Creating Model not implemented")
    raise Exception('Model not implemented.')

def main():
  if use_gpu == 'yes':
    board = LockGPU()
  else:
    print("Will go CPu then")
  model, train_op, eval_op = LoadExperiment(sys.argv[1], sys.argv[2],
                                            sys.argv[3])
  model = CreateDeepnet(model, train_op, eval_op)
  
  model.PrintNetwork()
  model.Train()
  
  model.PrintNetwork()
  if use_gpu == 'yes':
    FreeGPU(board)
  #raw_input('Press Enter.')

if __name__ == '__main__':
  main()
