from pyfft.cuda import Plan
import numpy as np
import matplotlib.pyplot as plt

import pycuda.driver as cuda
from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray

def nearest_2power(n):
    return np.power(2.,(np.ceil(np.log2(n))))

@profile
def main(context, stream, plan1, N1, N2):
    #N1 = # ffts applied
    #N2 = dim of ffts
    x = np.linspace(0, 2 * np.pi, N2)

    y = np.sin(2 * x)
    y = np.concatenate((y,np.zeros(nearest_2power(N2)-N2)))
    y = y.reshape(1,512)

    for i in xrange(49): #append 49 sines
        yi = np.sin(i * x)
        yi = np.concatenate((yi,np.zeros(nearest_2power(N2)-N2)))
        yi = yi.reshape(1,nearest_2power(N2))
        y = np.concatenate(((y),(yi)),0)
        
    y = y.transpose()
    yim= np.zeros(y.shape)
    y = np.array(y,np.float64)
    yw = y.transpose()
    yimw = yim.transpose()

    aw = np.fft.fft(yw,int(nearest_2power(N2)),1)
    bw = np.real(np.fft.ifft(aw,int(nearest_2power(N2)),1))

    '''    
    #plot numpy fft results
    f, axarr = plt.subplots(6, sharex=False)
    axarr[0].plot(y)
    axarr[0].set_title('input')
    axarr[1].plot(np.real(aw).transpose())
    axarr[1].set_title('output np.fft(input)')
    axarr[2].plot(bw.transpose())
    axarr[2].set_title('output np.ifft(np.fft(input))')
    '''

    gpu_testmat = gpuarray.to_gpu(y)
    gpu_testmatim = gpuarray.to_gpu(yim)
    plan1.execute(gpu_testmat, gpu_testmatim, batch=2) 
    c = gpu_testmat.get() #get fft result
    plan1.execute(gpu_testmat, gpu_testmatim, inverse=True, batch=2) 
    d = np.real(gpu_testmat.get()) #get ifft result
    
    '''
    #plot cuda fft results
    axarr[3].plot(y)
    axarr[3].set_title('input padded')
    axarr[4].plot(c)
    axarr[4].set_title('output Plan(input)')
    axarr[5].plot(d)
    axarr[5].set_title('output Plan(input, inverse=True)')
    plt.show()

    print "y= "
    print y
    print "c= "
    print c
    '''

#set dimensions
N1 = 50
N2 = 400
N = 1000

#init cuda
cuda.init()
context = make_default_context()
stream = cuda.Stream()
plan1 = Plan(int(nearest_2power(N2)), dtype=np.float64, context=context, stream=stream, fast_math=False)
#do operation N times for profiling purposes
for i in xrange(N):
    main(context=context, stream=stream, plan1=plan1, N1=N1, N2=N2)

#destroy cuda context
context.pop()
