import time, sys

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


def my_SOR(D, L, U, colsL, colsU, b, A, rank, size, error):
    """
    solving Ax = b to find x

    input parameters:
    error - maximum acceptable error
    A - csr matrix for fast norm calculation
    """
    n = 0
    privateN = 0
    w = 1.6
    myValues = []

    if rank == 0:
      n = len(D)

    n = comm.bcast(n, root = 0)    
    x = np.zeros(n)
    oldX = np.copy(x) 

    if rank == 0:     
      ranges = compute_range(n, size)
      for i in xrange(1, size):
        comm.send(ranges[i:i+2], dest = i, tag = i)
      myValues = [0]    
      myValues.append(ranges[1])      
    else:
      myValues = comm.recv(source = 0, tag = rank) 
    
    f = int(myValues[0])
    l = int(myValues[1])
    privateN = l-f

    colsL = comm.bcast(colsL, root = 0)
    colsU = comm.bcast(colsU, root = 0)
    L = comm.bcast(L, root = 0)
    U = comm.bcast(U, root = 0)
    b = comm.bcast(b, root = 0)
    A = comm.bcast(A, root = 0)
    

    if rank == 0:
      for i in xrange(1, size):
        comm.send(D[ranges[i]:ranges[i+1]], dest = i, tag = i)
    else:
      D = comm.recv(0, tag = rank)
    
    for iteration in xrange(100):
      s = np.zeros(privateN)
      for row in xrange(privateN):  

        for j in xrange(len(L[row+f])):
          s[row] += L[row+f][j] * x[colsL[row+f][j]]

        for j in xrange(len(U[row+f])):
          s[row] += U[row+f][j] * oldX[colsU[row+f][j]]

        x[row+f] += w * ((b[row+f]-s[row]) / D[row] - x[row+f])  
      if rank != 0:
          comm.send(x[f:l], dest = 0, tag = rank)
      else:
        for i in xrange(1, size):
          tmpx = comm.recv(source = i, tag = i)
          x[ ranges[i] : ranges[i+1]] = tmpx[0:len(tmpx)]       
      x = comm.bcast(x, root = 0)
      
      if my_residual(A, x, b) < error:
        break          
      oldX = np.copy(x)         
             
    
    if rank == 0:
      print "Error %f" % my_residual(A, x, b)  
      return x
    else:
      exit(0)


def my_residual(A,x,b):
  return np.linalg.norm(b-A.dot(x))

#returns non-zero elements and diagonal(1 row in values = 1 row in full matrix)
def organize_values(A, col, rows): 
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
    for j in xrange(col[i], col[i+1]):     
      if i < rows[j]:
        L[rows[j]].append(A[j])
        colsL[rows[j]].append(i)
      elif i > rows[j]:
        U[rows[j]].append(A[j])        
        colsU[rows[j]].append(i)
      else:
        D[rows[j]] = A[j]

  return (D,L,U,colsL,colsU)

#sets matrix ranges for each process
def compute_range(n, size):
  rangeList = np.zeros(size+1)
  elems = n/size
  rest = n%size
  j = 0
  for i in xrange(0,size):
    if i<rest:
      rangeList[i+1] = int((i+1)*elems+1+j)
      j += 1
    else:
      rangeList[i+1] = int((i+1)*elems+j)
  return rangeList


#-----------------------------------------------------------
#parsing input
#------------------------------------------------------------
if len(sys.argv) < 4:
    if rank == 0:
        print """Usage:
        mpiexec -np 2 python mpi_sor.py <matrix_filename> <vector_filename> <max_error>"""
    exit(0)

#-------------------------------------------------------------
#reading data from files
#-------------------------------------------------------------
if rank == 0:
  with open(sys.argv[1], 'r') as f:
    f.readline()
    val_line = f.readline()
    ind_line = f.readline()
    ptr_line = f.readline()

  dataA = np.fromstring(val_line[6:-2], sep = " ", dtype=float)
  indicesA = np.fromstring(ind_line[9:-2], sep = " ", dtype=int)
  indicesA -=1
  indptrA = np.fromstring(ptr_line[9:-2], sep = " ", dtype=int)
  indptrA -= 1

  #required for scipy csc_matrix format 
  indptrA = np.append(indptrA, len(dataA))

  with open(sys.argv[2], 'r') as f:
    f.readline()
    val_line = f.readline()
    ind_line = f.readline()

  dataB = np.fromstring(val_line[6:-3], sep = " ", dtype=float)
  indicesB = np.fromstring(ind_line[9:-2], sep = " ", dtype=int)
  indicesB -=1

  A = csc_matrix((dataA, indicesA, indptrA))
  A = A.tocsr() # used to check norm
  b = np.zeros(A.shape[0])
  b[indicesB] = dataB
  (D, L, U, colsL, colsU) = organize_values(dataA, indptrA, indicesA)
#initializing values for ranks!=0
else:
  D = []
  L = []
  U = []
  colsL = []
  colsU = []
  A = []
  b = []

#--------------------------------
#finally running SOR
#--------------------------------
x = my_SOR(D, L, U, colsL, colsU, b, A,rank, size, error=float(sys.argv[3]))

if rank == 0:
 with open('Xsolutions', 'w') as solX:
    for item in x:
        print >> solX, item
