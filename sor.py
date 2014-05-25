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


#SOLUTION BY SOR METHOD - bad approach
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

#SOLUTION BY SUCCESSIVE OVER RELAXATION METHOD
def run_SOR(A,b):
    """
    References: 
      http://en.wikipedia.org/wiki/Successive_over-relaxation
      http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    """
    x = np.zeros(b.shape[0])
    w = 1.2
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
        
    print "SOR: x=", x[120:125]
    print "norm = ", residual(A, x, b)


def residual(A, x, b):
   return  np.linalg.norm(b - A.dot(x))

def my_SOR(D,L,U,cols,b):
    x = np.zeros(b.shape[0])
    oldX = np.copy(x)
    w = 1.2
    n = len(D)
    for iteration in xrange(100):
      for i in xrange(n-1):
        s=np.zeros(n)
        
        for j in xrange(len(L[i])):
          s[i] = s[i] + L[i][j]*x[cols[i][j]]
        for j in xrange(len(U[i])):
          s[i] = s[i] + U[i][j]*oldX[cols[i][j]]
        s[i]=(b[i]-s[i])/D[i]
        oldX=np.copy(x)
        x[i]=oldX[i]+w*(s[i]-oldX[i])
    print x[120:125]
    return x

def organize_values(A,col,rows): #returns non-zero elements and diagonal(1 row in values = 1 row in full matrix)
  n = len(col)
  L = []
  U = []
  cols = []
  D = np.zeros(n)
  for i in xrange(n):
    L.append([])
    U.append([])
    cols.append([])

  for i in xrange(n-1):
    for j in xrange(col[i],col[i+1]):     
      if i<rows[j]:
        L[rows[j]].append(float(A[j]))
        cols[rows[j]].append(i)
      elif i>rows[j]:
        U[rows[j]].append(float(A[j]))
      else:
        D[rows[j]]=float(A[j])
        cols[rows[j]].append(i)
  #print values
  return (D,L,U,cols)

if __name__ == "__main__":

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
  (D,L,U,cols) = organize_values(dataA,indptrA,indicesA)
  #run_SOR_noob(A,b)
  run_SOR(A,b)
  x = my_SOR(D,L,U,cols,b)

  print residual(A,x,b)
  #with open('D.txt', 'w') as d:
  #  for item in D:
  #    print>>d, item
  #with open('L.txt', 'w') as l:
  #  for item in L:
  #    print>>l, item
  #with open('U.txt', 'w') as u:
  #  for item in U:
  #    print>>u, item
  #with open('cols.txt', 'w') as c:
  #  for item in cols:
  #    print>>c, item
  #run_exact(A,b)
