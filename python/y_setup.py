import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
from pprint import pprint as pp
from time import clock
import cPickle
import os
import os.path as path
from collections import OrderedDict
caffe.set_mode_cpu()
plt.rcParams['image.cmap']='gray'
def pit(o): pp(dir(o))

cat  ='../examples/images/cat.jpg'
mmean='caffe/imagenet/ilsvrc_2012_mean.npy'
INVal_path = '/home/noid/Downloads/INVal/'
grmimg_path= '/home/noid/grm/imgs/'
# a list mapping label_index to string name
IN_label_name = cPickle.load(open(INVal_path+"label_name.pkl"))
# a dict mapping filename to label_index
IN_img_label  = cPickle.load(open(INVal_path+"img_label.pkl"))
IN_img_label  = OrderedDict(sorted(IN_img_label.items(), key=lambda item: item[0]))

def checkimgs():
    g = gLenet
    for i in range(100):
        imgname = IN_img_label.keys()[i]
        g.s1(i)
        mpimg.imsave(
                grmimg_path+str(i)+"_data.jpg", 
                g.bt.get_data_img())
        for scale in [10,50,500, 2500]:
            mpimg.imsave(
                    grmimg_path+str(i)+"_x"+str(scale)+".jpg", 
                    g.bt.diffT*scale)



def get_label(index): return IN_img_label.items()[index][1]
def get_name (index): return IN_label_name[get_label(index)]
class INimg():
    def __init__(self, index):
        """
        returns fname(string), label(int), name(string)
            of the indexed file in ImageNet 2012 Val set
        """
        item = IN_img_label.items()[index]
        self.fname = INVal_path+"images/"+item[0]
        self.label = item[1]
        self.name  = IN_label_name[self.label]
    def load(self):
        return caffe.io.load_image(self.fname)
    def imshow(self):
        showimg(self.load())
    def __str__(self):
        return '{}: {}={}'.format(self.fname, self.label, self.name)


class Model():
    def __init__(self, specDict, weightFile):
        """
        The same model may use different specification train vs test
        So we have to specify an index when building the net
        """
        self.specDict = specDict
        self.weight= weightFile
        self.net = None
    def s0(self, index): # shortcut: build net, then run s1
        self.build()
        self.s1(index)
    def s1(self, index): # shortcut: set data, forward, backward, backtrace
        if self.net==None: self.build()
        self.setdata(INimg(index))
        self.bothpass()
        self.bt = BacktraceData(self.net)
    def s2(self, index, multiplier=[10,100], showM=False, useMult=False): # shortcut: set data, forward, backward, backtrace, show
        if self.net==None: self.build()
        self.s1(index)
        self.s3(index, multiplier, showM, useMult)
    def s3(self, index, multiplier=[10,100], showM=False, useMult=False):
        if self.net==None: self.build()
        self.bt.showdata()
        for scale in multiplier:
            data = self.bt.multT if useMult else self.bt.diffT
            showimg(data*scale)
        if showM:
            showimg(self.bt.diffM)
        pred_label = self.res['prob'].argmax()
        print "{}->{}".format(get_name(index), IN_label_name[pred_label])
        print "{}->{}".format(
                self.res['prob'][0,get_label(index)],
                self.res['prob'][0,pred_label])
    def build(self, key='deploy'):
        spec = self.specDict[key]
        self.net = caffe.Net( spec, self.weight, caffe.TEST)
        for k,v in self.net.blobs.items(): print k, v.data.shape
    def bothpass(self):
        self.res = self.net.forward()
        self.net.backward()
    def setdata(self, img=None, useMean=True, useScale=True, swap=False):
        """
        since each net may use different input size, we need to do some
        preprocessing. Right now it is unclear if we should do the mean and the
        scaling
        - mean: ImageNet has mean (104,0, 116.67, 122.67)
        - scaling: ImageNet has scale 255. Our cat has a scale of 1.
        - channel swap: CaffeNet uses BGR order instead of RGB. Not sure if the
          other nets also use BGR
        Problem: Transformer seems to always do mean subtraction before scaling.
        That really does not make sense if we subtract 1 by 104 then scale by
        255
        """
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        # load_image gives wrong order. Need to bring channel dim to first dim
        transformer.set_transpose('data', (2,0,1))
        if useMean:
            transformer.set_mean('data', np.load(mmean).mean(1).mean(1)) 
        if useScale:
            transformer.set_raw_scale('data', 255)  
        if swap:
            transformer.set_channel_swap('data', (2,1,0))  
        self.net.transformer = transformer

        if img==None:
            img_raw = caffe.io.load_image(cat)
            img_label= np.asarray([281])
        else:
            img_raw = img.load()
            img_label= np.asarray([img.label])
        img_data = transformer.preprocess('data', img_raw)
        img_data_contig=np.ascontiguousarray(img_data[np.newaxis, :])
        img_label_contig=img_label.astype(np.float32)

        if self.net.layers[0].type=="MemoryData":
            self.net.set_input_arrays(img_data_contig, img_label_contig)
        else: # assuming data of type 'Data'
            self.net.blobs['data'].data[...] = img_data
    def setMemDataZero(self):
        assert self.net.layers[0].type=='MemoryData', "only works with MemoryData layer"
        shape=self.net.blobs['data'].data.shape
        self.net._set_input_arrays(
                np.zeros(shape).astype(np.float32),
                np.zeros([shape[0],1,1,1]).astype(np.float32))
    def getBlobSize(self):
        size=0
        for k,v in self.net.blobs.items():
            size+=v.data.size
        return size
    def getWeightSize(self):
        size=0
        for layer in self.net.layers:
            for blob in layer.blobs:
                size+=blob.data.size
        return size
    def timepass(self):
        a=clock(); self.net.forward();b=clock();
        return b-a

CaffeNet = Model(
        { "deploy":'../models/bvlc_reference_caffenet/deploy.prototxt'
        , "train" :'../models/bvlc_reference_caffenet/train2.prototxt' 
        }
        ,          '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    )

NIN = Model(
        { "deploy":'../models/NINImageNet/y_deploy.prototxt' # yq's customized
        , "train" :'../models/NINImageNet/train_val.prototxt'
        } 
        ,          "../models/NINImageNet/nin_imagenet.caffemodel"
    )
gLenet = Model(
        { "deploy":"../models/bvlc_googlenet/deploy.prototxt"
        , "train" :"../models/bvlc_googlenet/train_val.prototxt"
        }
        ,          "../models/bvlc_googlenet/bvlc_googlenet.caffemodel"
    )

mine= Model(
        { "deploy":'../models/yq_fk1/backup_2_100000y/deploy.prototxt' # yq's customized
        , "train" :'../models/yq_fk1/backup_2_100000y/train_val.prototxt'
        } 
        ,          "../models/NINImageNet/nin_imagenet.caffemodel"
    )








def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    showimg(data)


def showwithz(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data, interpolation='nearest')

    numrows, numcols = data.shape
    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = data[row,col]
            return 'x=%1.4f, y=%1.4f, z=%4e'%(x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f'%(x, y)
    ax.format_coord = format_coord
    
    plt.show(block=False)

plot_args={'figsize':(5,5)}
def showimg(data): plt.figure(**plot_args); plt.imshow(data); plt.show(block=False)


def filterPower(arr, percent):
    sorted = np.sort(abs(arr.flatten()))[::-1]
    s = sum(sorted)
    cumsum=np.cumsum(sorted)
    threshold = percent * s
    idx = (cumsum<threshold).sum()-1
    abs_thresh = sorted[idx]
    return np.where(arr>abs_thresh, arr, np.zeros(arr.shape))


class BacktraceData():
    def __init__(self, net, swap=False):
        self.transformer = net.transformer
        self.data = np.copy(net.blobs['data'].data[0])
        if swap: self.data = self.data[[2,1,0],:,:]
        self.dataT= self.data.transpose(1,2,0)

        self.diff = np.copy(net.blobs['data'].diff[0])
        if swap: self.diff = self.diff[[2,1,0],:,:]
        self.diffT= self.diff.transpose(1,2,0)
        self.diffM= self.diff[0]+self.diff[1]+self.diff[2]

        self.mult = self.diff*self.data
        self.multT= self.mult[[2,1,0],:,:].transpose(1,2,0)
        self.multM= self.mult[0]+self.mult[1]+self.mult[2]
    def get_data_img(self):
        return self.transformer.deprocess('data', self.data)
    def showdata(self):
        showimg(self.get_data_img())

def doitall(net):
    net.forward()
    net.backward()
    dall = BacktraceData(net)
    net.blobs['fc8'].diff[0]=np.zeros([1000]).astype(np.float32)
    net.blobs['fc8'].diff[0,281] = -1
    net._backward(22,0)
    donly= BacktraceData(net)
    return dall, donly

"""
bvlc_alexnet
    t1: 57.1%
    t5: 80.2%

bvlc_googlenet
    t1: 68.7%
    t5: 88.9%

bvlc_reference_caffenet
    t1: 57.4%
    t5: 80.4%

NIN
    t1: 59.4%
    t5: unavailable

Observed problems for backtrace
- if we start backprop from softmax, gradient may be 0 when confidence is high
  See INVal 51 for example
- if we do not include softmax, we fail to eliminate the effects of competing
  classes. See kitten for example
- Easily enlarge "head region" for pets.
"""
