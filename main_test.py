from pyfft.cuda import Plan
import numpy as np
import matplotlib.pyplot as plt

import pycuda.driver as cuda
from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray

def nearest_2power(n):
    return np.power(2.,(np.ceil(np.log2(n))))

def main():
	testmat = np.ones((10, 10), dtype=np.complex64)
	print testmat

	padmat = np.zeros((10, int(nearest_2power(500)-10)))
	catestmat = np.concatenate((testmat,padmat),1)
	print testmat.shape, catestmat.shape

	a = np.fft.fft(catestmat,int(nearest_2power(2500)),1).transpose()
	b = np.real(np.fft.ifft(a.transpose(),int(nearest_2power(500)),1)).transpose()

	f, axarr = plt.subplots(6, sharex=False)
	axarr[0].plot(catestmat)
	axarr[0].set_title('input matrix')
	axarr[1].plot(a)
	axarr[1].set_title('output np.fft(input)')
	axarr[2].plot(b)
	axarr[2].set_title('output np.ifft(np.fft(input))')

#	print "a= "
#	print a
#	print "b= "
#	print b

	cuda.init()
	context = make_default_context()
	stream = cuda.Stream()

	plan = Plan((16, 16), stream=stream)

	gpu_testmat = gpuarray.to_gpu(catestmat)
	plan.execute(gpu_testmat) 
	c = gpu_testmat.get()
	plan.execute(gpu_testmat, inverse=True) 
	d = gpu_testmat.get()

	axarr[3].plot(catestmat)
	axarr[3].set_title('input padded')
	axarr[4].plot(c.transpose())
	axarr[4].set_title('output Plan(input)')
	axarr[5].plot(d)
	axarr[5].set_title('output Plan(input, inverse=True)')
	plt.show()

#	print "c= "
#	print c
#	print "d= "
#	print d

	context.pop()

main()
