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
def run_SOR(A, b):
    x= np.zeros(b.shape[0])
    w = 1.3
    n = b.shape[0]
    A=A.todense()
    #A=A.tocsr()
    print A.shape
    for iteration in xrange(20):
        for row in xrange(n):
            s = 0.0
            for col in xrange(n):
                if col!=row:
                    s = s + A[row,col] * x[col]
            if A[row,row] != 0.0:
                x[row] = x[row] + w*( (b[row]-s)/A[row,row] -x[row])

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

A = csc_matrix((dataA, indicesA, indptrA))
#print A
#print A.shape
#print A.todense()

b=np.array(dataB)
#b = csc_matrix((dataB, indicesB, indptrB), shape=(A.shape[0], 1), copy=True)
#print b
#print b.shape

run_SOR(A,b)