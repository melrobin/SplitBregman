# uncompyle6 version 3.0.0
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.13 (default, Dec  1 2017, 09:21:53) 
# [GCC 6.4.1 20170727 (Red Hat 6.4.1-1)]
# Embedded file name: /home/melrobin/research/ibraheem/SplitBregmanTVdenoising/split_bregman.py
# Compiled at: 2018-03-04 17:52:02
import numpy as np, pdb, scipy, scipy.sparse as sp, scipy.sparse.linalg as splinalg

def SB_ATV(g, mu):
    g = g.flatten()
    n = len(g)
    B, Bt, BtB = DiffOper(int(np.sqrt(n)))
    b = np.zeros((2 * n, 1))
    d = b
    u = g
    err = 1
    k = 1
    tol = 0.001
    lambda1 = 1
    while err > tol:
        print 'it. %d ' % k,
        up = u
        u, _ = splinalg.cg(sp.eye(n) + BtB, g - np.squeeze(lambda1 * Bt.dot(b - d)), tol=1e-05, maxiter=100)
        Bub = B * u + np.squeeze(b)
        print np.linalg.norm(Bub),
        d = np.maximum(np.abs(Bub) - mu / lambda1,0) * np.sign(Bub)
        b = Bub - d
        err = np.linalg.norm(up - u) / np.linalg.norm(u)
        print 'err=%g' % err
        k = k + 1

    print 'Stopped because norm(up-u)/norm(u) <= tol=%.1e\n', tol
    return u


def delete_row_lil(mat, i):
    if not isinstance(mat, scipy.sparse.lil_matrix):
        raise ValueError('works only for LIL format -- use .tolil() first')
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])


def delete_col_lil(mat, i):
    if not isinstance(mat, scipy.sparse.lil_matrix):
        raise ValueError('works only for LIL format -- use .tolil() first')
    mat.cols = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0], mat._shape[1] - 1)


def delete_row_csr(mat, i):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError('works only for CSR format -- use .tocsr() first')
    n = mat.indptr[i + 1] - mat.indptr[i]
    if n > 0:
        mat.data[(mat.indptr[i]):(-n)] = mat.data[mat.indptr[i + 1]:]
        mat.data = mat.data[:-n]
        mat.indices[(mat.indptr[i]):(-n)] = mat.indices[mat.indptr[i + 1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:(-1)] = mat.indptr[i + 1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0] - 1, mat._shape[1])


def DiffOper(N):
    data = np.vstack([-np.ones((1, N)), np.ones((1, N))])
    D = sp.diags(data, [0, 1], (N, N + 1), 'csr')
    print 'shape before: ', D.shape
    D = D[:, 1:]
    print 'shape afterward: ', D.shape
    D[(0, 0)] = 0
    print 'D dimensions: ', D.shape
    B = sp.vstack([sp.kron(sp.eye(N), D), sp.kron(D, sp.eye(N))], 'csr')
    Bt = B.transpose().tocsr()
    BtB = Bt * B
    print 'BtB dimensions: ', BtB.shape
    print 'Returned'
    return (
     B, Bt, BtB)
