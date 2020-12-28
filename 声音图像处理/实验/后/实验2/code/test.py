import numpy as np

a = np.array([[1,2],[3,4]])
print(a)
print(np.mean(a,axis=0))
print(np.mean(a,axis=1).reshape((-1,1)))

b = np.array([[1],[2],[3]])
c = np.array([[4],[5],[6]])
print(np.hstack([b,c]))