import numpy as np

"""EXACT SOLUTION
A * x = b
A^-1 * A * x = A^-1 * b
1 * x = A^-1 * b
"""
A = np.array([[ 1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  1.  ,  0.01,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.02,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.7 ,  0.  ,  1.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.5 ,  0.  ,  0.  ,  1.  ]])

b = np.array([1,2,3,4,5,6,7,8,9,10])
#print np.linalg.det(A)
x = np.dot(np.linalg.inv(A), b)

print "x=", x



"""SOLUTION BY SOR METHOD"""

x= np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
w = 1.6
n = len(x)
for iteration in xrange(100):
	for i in xrange(n):
		s = 0
		for j in xrange(n):
			if j!=i:
				s = s + A[i][j]*x[j]
	
		#x[i] = (1-w)*x[i] + w/A[i][i]*(b[i] - s)
		x[i] = x[i] + w*( (b[i]-s)/A[i][i] -x[i])

print "SOR: x=", x

#def residual(A, x, b):
#	return b - A*x

