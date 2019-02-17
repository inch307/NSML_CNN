import numpy as np
import math

a = np.array( [[1,2],[3,4]] )
b = np.array([[3,2],[7,2],[1,2]])

def get_euclidean_dist(q, r):
    result_mat = []
    for vec in q:
        sub = np.subtract(vec, r)
        q_mat = []
        for v in sub:
            sq = np.dot(v, v.T)
            sq = math.sqrt(sq)
            q_mat.append(sq)
        result_mat.append(q_mat)
    result_mat = np.array(result_mat)
    return result_mat

sim_matrix = get_euclidean_dist(a, b)
print(sim_matrix)
indices = np.argsort(sim_matrix, axis=1)
print(indices)
indices = np.flip(indices, axis=1)
print(indices)