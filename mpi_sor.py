import numpy as np
from scipy.sparse import csc_matrix
from mpi4py import MPI
#TODO: 
# -my_SOR parallel
# -fix bcast to send
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

def run_exact(A, b):
    A = A.todense()
    x = np.dot(np.linalg.inv(A), b)
    print "x=", x


def my_SOR(D,L,U,colsL,colsU,b,A,rank,size):
    n = 0
    privateN = 0
    w = 1.5
    myValues = []

    if rank == 0:
      n = len(D)

    n = comm.bcast(n,root=0)    
    x = np.zeros(n)
    oldX = np.copy(x) 

    if rank == 0:     
      ranges = compute_range(n,size)
      for i in xrange(1,size):
        comm.send(ranges[i:i+2],dest=i,tag=i)
      myValues=[0]    
      myValues.append(ranges[1])      
    else:
      myValues=comm.recv(0, tag=rank) 

    #print "Moj rank ", rank, " Wartosci " , myValues
    

    f=int(myValues[0])
    l=int(myValues[1])
    privateN = l-f
    
    colsL=comm.bcast(colsL,root=0)
    colsU=comm.bcast(colsU,root=0)
    L=comm.bcast(L,root=0)
    U=comm.bcast(U,root=0)


    if rank == 0:
      for i in xrange(1,size):
        comm.send(D[ranges[i]:ranges[i+1]],dest=i,tag=i)
        #c=comm.recv(source=1,tag=55)
        #comm.send(L[f:l+1],dest=i,tag=i*97)
        #comm.send(U[f:l+1],dest=i,tag=i*98)
        #comm.send(colsL[f:l+1],dest=i,tag=i+99)
        #comm.send(colsU[f:l+1],dest=i,tag=i+100)
    else:
      D=comm.recv(0,tag=rank)
      #sad=1
      #comm.send(sad,dest=0,tag=55)
      #print "sauuu"
      #L=comm.recv(0,tag=rank+97)
      #U=comm.recv(0,tag=rank+98)
      #colsL=comm.recv(0,tag=rank+99)
      #colsU=comm.recv(0,tag=rank+100)
    
    for iteration in xrange(100):
      #print iteration
      s=np.zeros(privateN)
      for row in xrange(privateN):  

        for j in xrange(len(L[row+f])):
          s[row] += L[row+f][j]*x[colsL[row+f][j]]

        for j in xrange(len(U[row+f])):
          s[row] += U[row+f][j]*oldX[colsU[row+f][j]]

        # tutaj kurwa epickie synchro
        if rank != 0:
          comm.send(x[f:l],dest=0,tag=rank)
        else:
          for i in xrange(1,size):
            #tutaj jest chujnia 
            print "i ",ranges[i]," ",ranges[i+1]
            x[ranges[i:i+1]]=comm.recv(source=i,tag=i)

        x=comm.bcast(x,root=0)
        #print "Moj tank",rank
        oldX=np.copy(x)
        #print "X ", len(x)
        x[row+f]+= w * ((b[row+f]-s[row]) / D[row+f] - x[row+f])
       
      
    if rank == 0:
      print my_residual(A,x,b)
      return x
    else:
      exit 


def my_residual(A,x,b):
  return np.linalg.norm(b-A.dot(x))

#returns non-zero elements and diagonal(1 row in values = 1 row in full matrix)
def organize_values(A,col,rows): 
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

#sets matrix ranges for each process
def compute_range(n,size):
  rangeList = np.zeros(size+1)
  elems = n/size
  rest = n%size
  j = 0
  for i in xrange(0,size):
    if i<rest:
      rangeList[i+1]=(i+1)*elems+1+j
      j += 1
    else:
      rangeList[i+1]=(i+1)*elems+j
  return rangeList


#reading data from files
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
#initializing values for ranks!=0
else:
  D = []
  L = []
  U = []
  colsL = []
  colsU = []
  A = []
  b = []


#SOR :)
x = my_SOR(D,L,U,colsL,colsU,b, A,rank,size)

#if rank==0:
# with open('Xsolution', 'w') as solX:
#    for item in x:
#        print>>solX, item
