import os
import time

def clean(ext):
    a = os.listdir('.')
    target     = [f for f in a if ext in f]
    target_num = [int(f.split('.')[0].split('_')[-1]) for f in target]
    target_rm  = [sum(map(lambda x:x>num, target_num))>=2  for num in target_num]
    for rm,f in zip(target_rm, target):
        if rm:
            print 'removing'+f
	    os.remove(f)
    

while True:
    clean('.solverstate')
    clean('.caffemodel')
    time.sleep(20)
