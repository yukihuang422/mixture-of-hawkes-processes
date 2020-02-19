import numpy as np
a = np.array([[0,1],
             [2,0],
             [3,1],
             [4,2],
             [5,1],
             [6,3],
             [7,0]])
seq = a[:,1].astype(int)
ahat = [0.1, 0.2, 0.3, 0.4]
au = np.array(ahat)[seq]
print(au)