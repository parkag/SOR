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
    A = A.todense()
    #print np.linalg.det(A)
    x = np.dot(np.linalg.inv(A), b)

    print "x=", x


"""SOLUTION BY SOR METHOD"""
def run_SOR_noob(A, b):
    x = np.zeros(b.shape[0])
    w = 1.5
    n = b.shape[0]
    A=A.todense()

    #rows,cols = A.nonzero()
    print A.shape
    for iteration in xrange(3):
        s=np.zeros(b.shape[0])
        for row in xrange(n):
            #s[row] = 0.0
            for col in xrange(n):
                if col!=row and A[row,col] != 0.0:
                    s[row] += A[row,col] * x[col]
            if A[row,row] != 0.0:
                x[row] += w*( (b[row]-s[row])/A[row,row] -x[row])
        print "s=", s

    print "SOR: x=", x
    print "norm = ", residual_dense(A, x, b)

def run_SOR(A,b):
    x = np.zeros(b.shape[0])
    w = 1.5
    cols = b.shape[0]
    D = A.diagonal()

    for iteration in xrange(3):
        s = np.zeros(b.shape[0])  

        for i in xrange(cols):
            rows = A.indices[A.indptr[i]:A.indptr[i+1]]
            s[rows] += A.data[rows] * x[i]
            s[i] -= x[i]*D[i]

            if D[i]!=0.0:
              x[i] += w*( (b[i]-s[i])/D[i] -x[i])
        #s[cols-1] = x[cols-1]*D[cols-1]
        """for row in xrange(cols):
            if D[row]!=0.0:
              x[row] += w*( (b[row]-s[row])/D[row] -x[row])"""
        print "s=", s
    print "SOR: x=", x
    print "norm = ", residual(A, x, b)


def residual(A, x, b):
   A=A.todense()
   return  np.linalg.norm(b - np.dot(A,x))

def residual_dense(A,x,b):
  return np.linalg.norm(b - np.dot(A,x))

def organize_values(A,col,rows): #returns non-zero elements and diagonal(1 row in values = 1 row in full matrix)
  n = len(col)
  values = []
  
  for i in xrange(n):
    values.append([])
  diagonal = np.zeros(n)

  for i in xrange(n-1):
    for j in xrange(col[i],col[i+1]):     
      if i!=rows[j]:
        values[rows[j]].append(float(A[j]))
      else:
        diagonal[rows[j]]=float(A[j])
  #print diagonal
  #print values
  return (values,diagonal)

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
b = np.array(dataB)
(vals,diag)=organize_values(dataA,indptrA,indicesA) 

#run_SOR_noob(A,b)
#run_SOR(A,b)


#run_exact(A,b)
