from pyfft.cuda import Plan
import numpy as np
import traceback
import pickle
import sys
from math import ceil

testmat = np.identity(5, "double") #I may or may not be able to find how to fill a matrix with a constant in numpy
testmat = (testmat/float("inf"))+20
print testmat

a = np.fft.fft(testmat, 8, -1)
b = np.real(np.fft.ifft(a, 8, -1))

agp =

print "a= "
print a
print "b= "
print b