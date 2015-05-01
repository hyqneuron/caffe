import sys
sys.path.append('python')
import caffe
flickrsolver = 'models/yq_test1/solver.prototxt'
def getsolver(solver=flickrsolver):
    return caffe.get_solver(solver)
