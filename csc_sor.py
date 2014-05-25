import numpy as np
from scipy.sparse import csc_matrix
from matplotlib import pylab as plt

"""
EXACT SOLUTION

A * x = b

A^-1 * A * x = A^-1 * b

1 * x = A^-1 * b
"""
def run_exact(A, b):
    A = np.array()
    #print np.linalg.det(A)
    x = np.dot(np.linalg.inv(A), b)

    print "x=", x


"""SOLUTION BY SOR METHOD"""
def run_SOR(A, b, col, row):
    n = b.shape[0]
    x= np.zeros(n)
    res = np.zeros(n)
    s = np.zeros(n)
    w = 1.3
    j = 1
    for iteration in xrange(20):
        for i in range(1,len(col)):
	    current = 0
            while i!=col[i]:
                current = row[j]
		if i!=current:
                    s[current] = s[current]# + A[j-1] * x[current]
		else:
		    res[current] = A[j-1]
		j = j+1
            if res[i] != 0.0:
                x[current] = x[current] + w*( (b[current]-s[current])/res[current] -x[current])

    print "SOR: x=", x
    print "norm = ", residual(A, x, b)

def residual(A, x, b):
   return  np.linalg.norm(b - np.dot(A,x))

with open('data/matrixA.dat', 'r') as f:
  f.readline()
  val_line = f.readline()
  ind_line = f.readline()
  ptr_line = f.readline()

dataA = np.fromstring(val_line[6:-2], sep = " ", dtype=float)
indicesA =  np.fromstring(ind_line[9:-2], sep = " ", dtype=int)
indicesA -=1
indptrA = np.fromstring(ptr_line[9:-2], sep = " ", dtype=int)
indptrA -= 1

#format required by scipy
#http://scipy-lectures.github.io/advanced/scipy_sparse/csc_matrix.html
indptrA= np.append(indptrA, len(dataA))

#print indptrA.shape

with open('data/vectorB.dat', 'r') as f:
  f.readline()
  val_line = f.readline()
  ind_line = f.readline()
  ptr_line = f.readline()

dataB = np.fromstring(val_line[6:-3], sep = " ", dtype=float)
indicesB =  np.fromstring(ind_line[9:-3], sep = " ", dtype=int)
indicesB -=1
indptrB = np.fromstring(ptr_line[9:-3], sep = " ", dtype=int)
indptrB -= 1

A = np.array(dataA)

#print A
#print A.shape
#print A.todense()

b=np.array(dataB)
col = np.array(indicesA)
row = np.array(indptrA)
#b = csc_matrix((dataB, indicesB, indptrB), shape=(A.shape[0], 1), copy=True)
#print b
#print b.shape
print len(row)
print len(col)
print  row
print  col
print A
run_SOR(A,b,col,row)
