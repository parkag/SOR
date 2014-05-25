import numpy as np
from scipy.sparse import csc_matrix

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

#SOLUTION BY SUCCESSIVE OVER RELAXATION METHOD
def run_SOR(A,b):
    """
    References: 
      http://en.wikipedia.org/wiki/Successive_over-relaxation
      http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    """
    x = np.zeros(b.shape[0])
    w = 1.6
    rows = b.shape[0]
    D = A.diagonal()
    A = A.tocsr()
    
    for iteration in xrange(100):
        s = np.zeros(b.shape[0])  
        for row in xrange(rows):
            cols = A.indices[A.indptr[row]:A.indptr[row+1]]
            
            for j, col in enumerate(cols):
                if col!=row:
                    s[row] += A.data[A.indptr[row]:A.indptr[row+1]][j]*x[col]

            if D[row]!=0.0:
                x[row] += w*( (b[row]-s[row])/D[row] - x[row])
        
    print "SOR: x=", x
    print "norm = ", residual(A, x, b)


def residual(A, x, b):
   return  np.linalg.norm(b - A.dot(x))


with open('data/matrixA.dat', 'r') as f:
  f.readline()
  val_line = f.readline()
  ind_line = f.readline()
  ptr_line = f.readline()

dataA = np.fromstring(val_line[6:-3], sep = " ", dtype=float)
indicesA =  np.fromstring(ind_line[9:-3], sep = " ", dtype=int)
indicesA -=1
indptrA = np.fromstring(ptr_line[9:-3], sep = " ", dtype=int)
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

run_SOR(A,b)