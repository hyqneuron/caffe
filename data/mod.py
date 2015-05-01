
ftrain = open('train.txt', 'r')
ftest  = open( 'test.txt', 'r')
fmtrain= open('train.mod.txt','w')
fmtest = open( 'test.mod.txt','w')

for line in ftrain:
    segs = line.split()
    fmtrain.write(' '.join([segs[0], segs[1], segs[1]]))
    fmtrain.write('\n')
for line in ftest:
    segs = line.split()
    fmtest.write(' '.join([segs[0], segs[1], segs[1]]))
    fmtest.write('\n')

ftrain.close()
ftest.close()
fmtrain.close()
fmtest.close()
