from pyfft.cuda import Plan
import numpy as np
import matplotlib.pyplot as plt

import pycuda.driver as cuda
from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray

def nearest_2power(n):
    return np.power(2.,(np.ceil(np.log2(n))))

def main():
	testmat = np.ones((100, 1), dtype=np.complex64)
#	print testmat

	x = np.linspace(0, 2 * np.pi, 400)
	y1 = np.sin(2 * x)
	y1 = np.concatenate((y1,np.zeros(112)))
	y1 = y1.reshape(1,512)
	y2 = np.sin(5 * x)
	y2 = np.concatenate((y2,np.zeros(112)))
	y2 = y2.reshape(1,512)
	y = np.concatenate(((y1),(y2)),0).transpose()
	yim= np.zeros((2,512))

	a = np.fft.fft(y,int(nearest_2power(400)),0)
	b = np.real(np.fft.ifft(a,int(nearest_2power(400)),0))

	f, axarr = plt.subplots(6, sharex=False)
	axarr[0].plot(y)
	axarr[0].set_title('input')
	axarr[1].plot(a)
	axarr[1].set_title('output np.fft(input)')
	axarr[2].plot(b)
	axarr[2].set_title('output np.ifft(np.fft(input))')

#	print "a= "
#	print a.transpose()
#	print "b= "
#	print b.transpose()

	cuda.init()
	context = make_default_context()
	stream = cuda.Stream()

	plan = Plan((512,2), dtype=np.float64, context=context, stream=stream, fast_math=False)

	gpu_testmat = gpuarray.to_gpu(y)
	gpu_testmatim = gpuarray.to_gpu(yim)
	plan.execute(gpu_testmat, gpu_testmatim, batch=1) 
	c = gpu_testmat.get()
	plan.execute(gpu_testmat, gpu_testmatim, inverse=True, batch=1) 
	d = np.real(gpu_testmat.get())

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
#	print "d= "
#	print d

	context.pop()

main()
