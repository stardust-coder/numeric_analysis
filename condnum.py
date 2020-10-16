import numpy as np

mat = np.array([[0.780,0.563],[0.913,0.659]])
cond = np.linalg.cond(mat)
print(cond)
