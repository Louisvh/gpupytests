from pyfft.cuda import Plan
import numpy as np

import pycuda.driver as cuda
from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray

testmat = np.ones((50, 50), dtype=np.complex64)
print testmat

a = np.fft.fft(testmat, 128, -1)
b = np.real(np.fft.ifft(a, 128, -1))

print "a= "
print a
print "b= "
print b

cuda.init()
context = make_default_context()
stream = cuda.Stream()

plan = Plan((16, 16), stream=stream)

gpu_testmat = gpuarray.to_gpu(data)
plan.execute(gpu_data) 
c = gpu_data.get()
print result 

context.pop()
