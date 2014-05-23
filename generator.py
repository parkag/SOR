import scipy.sparse

A = scipy.sparse.rand(10000, 10000, density = 0.01)
print A.tocsr()
#print A
