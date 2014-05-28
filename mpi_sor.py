import numpy as np
from scipy.sparse import csc_matrix
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

def run_exact(A, b):
    A = A.todense()
    x = np.dot(np.linalg.inv(A), b)
    print "x=", x


def my_SOR(D,L,U,colsL,colsU,b,A,rank,size):
    n = 0
    if rank == 0:
      n = len(D)

    n = comm.bcast(n,root=0)

    print n 
    if rank == 0:
      x = np.zeros(n)
      oldX = np.copy(x)
      w = 1.5

      #
      for iteration in xrange(100):
        s=np.zeros(n)
        for row in xrange(n-1):  

          for j in xrange(len(L[row])):
            s[row] += L[row][j]*x[colsL[row][j]]

          for j in xrange(len(U[row])):
            s[row] += U[row][j]*oldX[colsU[row][j]]

          oldX=np.copy(x)
          x[row]+= w * ((b[row]-s[row]) / D[row] - x[row])
      #
    if rank == 0:
      print my_residual(A,x,b)
      return x
    else:
      return [] 


def my_residual(A,x,b):
  return np.linalg.norm(b-A.dot(x))-2

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



###############################################
if rank==0:
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

  indptrA= np.append(indptrA, len(dataA))

  with open('data/vectorB.dat', 'r') as f:
    f.readline()
    val_line = f.readline()

  dataB = np.fromstring(val_line[6:-3], sep = " ", dtype=float)

  A = csc_matrix((dataA, indicesA, indptrA))
  A = A.tocsr() # used to check norm
  b = np.array(dataB)
  (D,L,U,colsL,colsU) = organize_values(dataA,indptrA,indicesA)
  ##################################################
else:
  D = []
  L = []
  U = []
  colsL = []
  colsU = []
  A = []
  b = []


#tutaj broadcast/send D,L,U,cols...


#


x = my_SOR(D,L,U,colsL,colsU,b, A,rank,size)

#if rank==0:
# with open('Xsolution', 'w') as solX:
#    for item in x:
#        print>>solX, item
