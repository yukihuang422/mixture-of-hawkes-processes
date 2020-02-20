import numpy as np
from sklearn.utils.extmath import cartesian
a = np.array([[0,0],
             [2,0],
             [3,1],
             [4,1],
             [5,3],
             [6,3],
             [7,2]])
seq = a[:,1].astype(int)
ahat = [0.1, 0.2, 0.3, 0.4]
au = np.array(ahat)[seq]
c = cartesian([np.where(a[:,1]==1)[0], np.where(a[:,1]==0)[0]])
print(c)
print(a[c[:,0],c[:,1]])