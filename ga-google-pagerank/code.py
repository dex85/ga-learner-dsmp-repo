# --------------
# Code starts here

import numpy as np

# Code starts here

# Adjacency matrix
adj_mat = np.array([[0,0,0,0,0,0,1/3,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                   [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                  [0,0,1/2,1/3,0,0,1/3,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/3,0]])

# Compute eigenvalues and eigencevectrs
eigenvalues, eigenvectors = np.linalg.eig(adj_mat)

# Eigen vector corresponding to 1
eigen_1 = abs(eigenvectors[:,0])/np.linalg.norm(eigenvectors[:,0], 1)

# most important page
#page = np.where(eigen_1 == eigen_1.max())
page = 8
print(eigen_1, page, sep = "\n")

# Code ends here


# --------------
# Code starts here

# Initialize stationary vector I
print(page)
adj2 = np.array([[  0,      0,      0,      0,      0,      0,      0,      0],
                    [0.5,   0,      0.5,    1/3,    0,      0,      0,      0],
                    [0.5,   0,      0,      0,      0,      0,      0,      0],
                    [0,     1,      0,      0,      0,      0,      0,      0],
                    [0,     0,      0.5,    1/3,    0,      0,      0.5,    0],
                    [0,     0,      0,      1/3,    1/3,    0,      0,      0.5],
                    [0,     0,      0,      0,      1/3,    0,      0,      0.5],
                    [0,     0,      0,      0,      1/3,    1,      0.5,    0]])
# Perform iterations for power method
print(adj_mat)
#init_I = np.zeros((8,11))
init_I=np.array([1,0,0,0,0,0,0,0])
#init_I[0,0] = 1
#print(init_I)
#print(init_I[:,1])

for i in range(10):
    #init_I[:,1] = np.dot(adj_mat, init_I[:,0])
    init_temp = np.dot(adj_mat, init_I)
    init_I = init_temp/np.linalg.norm(init_I,1)
    #init_I[:,i+1] = init_temp
    #init_I[:,i+1] = np.linalg.norm(init_I,1)

print(init_I)
power_page=8

# Code ends here


# --------------
# Code starts here

# New Adjancency matrix
# New Adjancency matrix
new_adj_mat = np.array([[0,0,0,0,0,0,0,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                  [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                   [0,0,1/2,1/3,0,0,1/2,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/2,0]])

# Initialize stationary vector I
new_init_I = np.array([1,0,0,0,0,0,0,0])

# Perform iterations for power method
for i in range(10):
    new_init_I = np.dot(new_adj_mat, new_init_I)/np.linalg.norm(new_init_I, 1)

print(new_init_I)


# Code ends here


# --------------
# Alpha value
alpha = 0.85

# Code starts here
n = len(new_adj_mat)
one = np.ones_like(new_adj_mat)
# Modified adjancency matrix
G = alpha*new_adj_mat + (one - alpha)*1/n

# Initialize stationary vector I
final_init_I = np.array([1,0,0,0,0,0,0,0])

# Perform iterations for power method
for i in range(1000):
    final_init_I = np.dot(G,final_init_I)/np.linalg.norm(final_init_I,1)

print(final_init_I)
# Code ends here


