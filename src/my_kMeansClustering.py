import numpy as np


def my_kMeansClustering(X, k, initialCentres, maxIter=500):
    # Input
    # X : N-by-D matrix (double) of input sample data
    # k : scalar (integer) - the number of clusters
    # initialCentres : k-by-D matrix (double) of initial cluster centres
    # maxIter  : scalar (integer) - the maximum number of iterations
    # Output
    # C   : k-by-D matrix (double) of cluster centres
    # idx : N-by-1 vector (integer) of cluster index table
    # SSE : (L+1)-by-1 vector (double) of sum-squared-errors
    
    C = []
    idx = []
    SSE = []


    N = len(A)
    dim = A[0].size

    D = np.zeros((K, N))

    idx_prev = np.empty_like(D.argmin(axis=0))

    # initialise error list
    SSE = []

    # show cluster centres at iteration 0
    print("[0] Iteration: ", centres.tolist())

    # Compute Squared Euclidean distance (i.e. the squared distance)
    # between each cluster centre and each observation
    for i in range(maxiter):
        for c in range(K):
            D[c] = sq_dist(A, centres[:, c])

        # Assign data to clusters
        # Ds are the actual distances and idx are the cluster assignments
        idx = D.argmin(axis=0)  # find min dist. for each observation

        # Update cluster centres
        for c in range(K):
            # check the number of samples assigned to this cluster
            if (np.sum(idx == c) == 0):
                print(f'k-means: cluster {c} is empty')
            else:
                errors.append(sum_sq_error(A, K, N, centres, idx))
                centres[c] = np.mean(A[idx[:] == c], axis=0)

        if (out):  # show cluster centres at iteration i
            print(f'[{i + 1}] Iteration: ', centres.tolist())

            # If assignments were maintained,
        if (np.array_equal(idx, idx_prev)):
            return centres, errors

        # Store current assignment to check if it changes in next iteration
        idx_prev = idx

    return C, idx, SSE


def sq_dist(U, v):
    return np.sum((U-v)**2, axis=1)


def sum_sq_error(A, K, N, centres, idx):
    z = np.zeros((K, N), dtype=bool)
    error = 0

    for row in range(K):
        for col in range(N):
            if (idx[col] == row):
                z[row, col] = True

    for cluster in range(K):
        error += np.sum(sq_dist(A[z[cluster, :]], centres[cluster]))

    error *= (1 / N)

    return error