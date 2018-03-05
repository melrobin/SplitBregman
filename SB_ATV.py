# uncompyle6 version 3.0.0
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.13 (default, Dec  1 2017, 09:21:53) 
# [GCC 6.4.1 20170727 (Red Hat 6.4.1-1)]
# Embedded file name: /home/melrobin/research/ibraheem/SplitBregmanTVdenoising/SB_ATV.py
# Compiled at: 2017-08-06 21:39:22
import numpy as np, scipy.sparse as sp

def SB_ATV(g, mu):
    g = g.flatten()
    n = len(g)
    B, Bt, BtB = DiffOper(sqrt(n))
    b = np.zeros(2 * n, 1)
    d = b
    u = g
    err = 1
    k = 1
    tol = 0.001
    lambda1 = 1
    while err > tol:
        print 'it. %g ', k
        up = u
        u, _ = sp.linalg.cg(sp.eye(n) + BtB, g - lambda1 * Bt * (b - d), tol=1e-05, maxiter=100)
        Bub = B * u + b
        d = max(np.abs(Bub) - mu / lambda1, 0) * np.sign(Bub)
        b = Bub - d
        err = np.linalg.norm(up - u) / np.linalg.norm(u)
        print 'err=%g \n', err
        k = k + 1

    print 'Stopped because norm(up-u)/norm(u) <= tol=%.1e\n', tol
    return u


def DiffOper(N):
    D = sp.diags([-np.ones(N, 1), np.ones(N, 1)], [0, 1], N, N + 1)
    D[:, 1] = []
    D[(1, 1)] = 0
    B = [[np.kron(sp.eye(N), D)], [np.kron(D, sp.eye(N))]]
    Bt = np.tranpose(B)
    BtB = Bt * B
    return (
     B, Bt, BtB)
