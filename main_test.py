from pyfft.cuda import Plan
import numpy as np

import pycuda.driver as cuda
from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray

def nearest_2power(n):
    return np.power(2.,(np.ceil(np.log2(n))))


@profile
def main():
	testmat = np.ones((50, 50), dtype=np.complex64)
	#print testmat

	a = np.fft.fft(testmat,int(nearest_2power(2500)),-1)
	b = np.real(np.fft.ifft(a,int(nearest_2power(2500)),-1))

#	print "a= "
#	print a
#	print "b= "
#	print b

	cuda.init()
	context = make_default_context()
	stream = cuda.Stream()

	plan = Plan((16, 16), stream=stream)

	gpu_testmat = gpuarray.to_gpu(testmat)
	plan.execute(gpu_testmat) 
	c = gpu_testmat.get()

	plan.execute(gpu_testmat, inverse=True) 
	d = gpu_testmat.get()

#	print "c= "
#	print c
#	print "d= "
#	print d

	context.pop()

main()
