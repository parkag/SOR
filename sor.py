import numpy as np
from scipy.sparse import csc_matrix
from matplotlib import pylab as plt
import sys
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
def run_SOR(A,b,error):
    """
    References: 
      http://en.wikipedia.org/wiki/Successive_over-relaxation
      http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    """
    x = np.zeros(b.shape[0])
    w_0 = 1.5
    rows = b.shape[0]
    D = A.diagonal()
    A = A.tocsr()
    iterations = 0
    #err = 100.0

    w_prev_prev = w_0 + 0.1
    w_prev = w_0 + 0.05
    w = w_0

    err = my_residual(A,x,b)
    err_prev = err + 1.0
    while err > error:
        s = np.zeros(b.shape[0])  
        for row in xrange(rows):
            cols = A.indices[A.indptr[row]:A.indptr[row+1]]
            
            for j, col in enumerate(cols):
                if col!=row:
                    s[row] += A.data[A.indptr[row]:A.indptr[row+1]][j]*x[col]

            if D[row]!=0.0:
                x[row] += w*( (b[row]-s[row])/D[row] - x[row])
        iterations += 1

        err_prev_prev = err_prev
        err_prev = err
        err = my_residual(A,x,b)
        print "omega = ", w
        print "prev_omega = ", w_prev
        print "norm(error): ", err
        print "prev_norm: ", err_prev
        if iterations%2 == 0:
          [w, w_prev, w_prev_prev] = update_omega(err, err_prev, err_prev_prev, w, w_prev, w_prev_prev)

    print "SOR: x=", x
    print "norm = ", my_residual(A, x, b)
    print "iterations:", iterations

def update_omega(norm, prev_norm, prev_prev_norm, omega, prev_omega, prev_prev_omega):
    """ 
    References:
    http://www.researchgate.net/publication/2651797_Adaptive_SOR_A_case_study_in_automatic_differentiation_of_algorithm_parameters
    """
    d_om = omega - prev_omega
    d_om2 = prev_omega - prev_prev_omega
    d_err = norm - prev_norm
    d_err2 = prev_norm - prev_prev_norm
    d_omega = min(0.05, (d_err/d_om)*(omega - prev_omega)/(d_err/d_om - d_err2/d_om2 ))
    print d_omega
    prev_omega = omega
    omega = max(0.5, min(1.985, omega+d_omega))
    return [omega, prev_omega, prev_prev_omega]

def my_SOR(D,L,U,colsL,colsU,b,A):
    x = np.zeros(b.shape[0])
    oldX = np.copy(x)
    w = 1.9
    n = len(D)
    A=A.tocsr()
    for iteration in xrange(100):
      s=np.zeros(n)
      for row in xrange(n-1):  

        for j in xrange(len(L[row])):
          s[row] += L[row][j]*x[colsL[row][j]]

        for j in xrange(len(U[row])):
          s[row] += U[row][j]*oldX[colsU[row][j]]

        oldX=np.copy(x)
        x[row]+= w * ((b[row]-s[row]) / D[row] - x[row])
      print my_residual(A,x,b)

    return x


def my_residual(A,x,b):
  return np.linalg.norm(b-A.dot(x))

def residual(A, x, b):
   A=A.todense()
   return  np.linalg.norm(b - np.dot(A,x))

def residual_dense(A,x,b):
  return np.linalg.norm(b - np.dot(A,x))

def organize_values(A,col,rows): #returns non-zero elements and diagonal(1 row in values = 1 row in full matrix)
  n = len(col)
  L = []
  U = []
  colsL = []
  colsU = []
  D = np.zeros(n-1)
  for i in xrange(n-1):
    L.append([])
    U.append([])
    colsL.append([])
    colsU.append([])

  for i in xrange(n-1):
    for j in xrange(col[i],col[i+1]):     
      if i<rows[j]:
        L[rows[j]].append(A[j])
        colsL[rows[j]].append(i)
      elif i>rows[j]:
        U[rows[j]].append(A[j])        
        colsU[rows[j]].append(i)
      else:
        D[rows[j]]=A[j]
  
  return (D,L,U,colsL,colsU)

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
  #ind_line = f.readline()
  #ptr_line = f.readline()

dataB = np.fromstring(val_line[6:-3], sep = " ", dtype=float)
#indicesB =  np.fromstring(ind_line[9:-3], sep = " ", dtype=int)
#indicesB -=1
#indptrB = np.fromstring(ptr_line[9:-3], sep = " ", dtype=int)
#indptrB -= 1

A = csc_matrix((dataA, indicesA, indptrA))
b = np.array(dataB)
#(D,L,U,colsL,colsU) = organize_values(dataA,indptrA,indicesA)


#x = my_SOR(D,L,U,colsL,colsU,b, A)
print "DP/\---------------------------\/GP"
run_SOR(A,b, error=.000001)


#with open('Xsolution', 'w') as solX:
#  for item in x:
#      print>>solX, item
#with open('D.txt', 'w') as d:
#  for item in D:
#    print>>d, item
#run_exact(A,b)
