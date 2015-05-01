import sys
sys.path.append('python')
import caffe
flickrsolver = 'models/yq_test1/solver.prototxt'
caffenetmodel= 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
def getsolver(solver=flickrsolver):
    return caffe.get_solver(solver)
def copyweights(solver):
    solver.net.copy_from(caffenetmodel)
    for net in solver.test_nets:
        net.copy_from(caffenetmodel)
