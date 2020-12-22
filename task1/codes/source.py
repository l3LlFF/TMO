import numpy as np
import sys

def antidurak(D_1):
    durak = False
    e=np.finfo(float).eps
    for x in D_1.sum(axis=1):
        durak = durak or (x > e)
    for x in D_1.diagonal().tolist()[0]:
        durak = durak or (x > 0)
    return bool(durak)

def find_theta(matr, e):
    D_1e = np.concatenate((matr.transpose(), e.transpose()))
    c = np.concatenate((np.zeros([n,1]), np.matrix(1)))
    theta = np.linalg.lstsq(a = D_1e, b = c,rcond=None)
    theta = theta[0].reshape((1, n))
    return theta

def find_degenerative(T):
    b=[]
    for i in range(4):
        b.append(T**(i+1))
    return b

def find_b(T1, T2):
    b = []

    for i in range(1, 5):
        b.append((T2**(i + 1) - T1**(i + 1)) / ((i+1) * (T2 - T1)))
    return b 

def find_dD(D, D_1):
    dD = []
    dD.append(D_1)
    for i in range(4):
        dD.append(np.matrix(np.zeros([n,n])))
    for index, matrix in enumerate(D):
        dD[1] += index * matrix
        dD[2] += index * (index - 1) * matrix
        dD[3] += index * (index - 1) * (index - 2) * matrix
        dD[4] += index * (index - 1) * (index - 2) * (index - 3) * matrix
    return dD

def find_A(d_mtrx, b):
    A = []
    A.append(d_mtrx[0])
    A.append(-d_mtrx[1] * b[0])
    A.append((d_mtrx[2] * b[0]**2 + d_mtrx[1] * b[1])/2)
    A.append(-(d_mtrx[3] * b[0]**3 + 3 * d_mtrx[2] * b[0] * b[1] + d_mtrx[1] * b[2])/6)
    A.append((d_mtrx[4] * b[0]**4 + 6 * d_mtrx[3] * b[0]**2 * b[1] + 4 * d_mtrx[2] * b[0] * b[2] + 
              3 * d_mtrx[2] * b[1]**2 + d_mtrx[1] * b[3])/24)
    return A

def find_wr(mtrx, b, lmbd, pi0, I_, I, A, A_):
    p0 = lmbd * pi0 * np.linalg.inv(-mtrx[0])
    w = []
    w.append(theta)
    A_inv = np.linalg.inv(A_)
    w.append(((w[0] * (I + A[1]) - p0) * I_ + w[0] * A[2] * e * e_) * A_inv)
    w.append(-2 * ((w[0] * A[2] - w[1] * (I + A[1])) * I_ + (w[0] * A[3] - w[1] * A[2]) * e * e_) * A_inv)
    w.append(3 * ((2 * w[0] * A[3] - 2 * w[1] * A[2] + w[2] * (I + A[1])) * I_ + 
     (2 * w[0] * A[4] - 2 * w[1] * A[3] + w[2] * A[2]) * e * e_) * A_inv)
    return w

def find_v(w, A, lmbd, b, rho):
    v = []
    v.append(np.matrix([1]))
    v.append(((w[0] * A[2] - w[1] * A[1]) * e - lmbd * b[1] / 2)/rho)
    v.append(((-2 * w[0] * A[3] + 2 * w[1] * A[2] - w[2] * A[1]) * e -
              v[0] * lmbd * b[2] / 3 - v[1] * lmbd * b[1]) / rho) 
    v.append(((6 * w[0] * A[4] - 6 * w[1] * A[3] + 3 * w[2] * A[2] - w[3] * A[1]) * e - 
              lmbd * (v[0] * b[3] / 4 + v[1] * b[2] + 3 * v[2] * b[1] / 2)) / rho)

    return v

def find_rho(lmbd, b):
    return lmbd * b[0]

if __name__ == "__main__":
    task = open("task.txt", "r")
    n = int(task.readline())
    k = int(task.readline())
    D = []

    for j in range(k):
        line = task.readline()
        D.append(np.matrix(line))

    T1 = float(task.readline())
    T2 = float(task.readline())
    pi0 = np.matrix(task.readline())
    e = np.full([n,1], 1)
    e_ = np.full([n,1], 0)
    e_[0] = 1

    I_ = np.matrix(np.diag((e-e_).transpose()[0]))
    I = np.matrix(np.diag(e.transpose()[0]))

    e_ = e_.transpose()

    D_1 = sum(D)
    if(antidurak(D_1)):
        sys.exit("D(1) isn't a generator")

    theta = find_theta(D_1, e)

    dD = []

    dD.append(D_1)

    dD = find_dD(D, D_1)

    b = find_b(T1, T2)
    lambda_ = theta * dD[1] * e

    A = find_A(dD, b)

    A_ = A[0] * I_ + (I + A[1]) * e * e_

    w = find_wr(D, b, lambda_, pi0, I_, I, A, A_)

    rho = find_rho(lambda_, b)

    v = find_v(w, A, lambda_, b, rho)

    print(v)
    print(w)
