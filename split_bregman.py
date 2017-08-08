import numpy as np
import pdb
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
'''
 Split Bregman Anisotropic Total Variation Denoising

   u = arg min_u 1/2||u-g||_2^2 + mu*ATV(u)
   
   g : noisy image
   mu: regularisation parameter
   u : denoised image

 Refs:
  *Goldstein and Osher, The split Bregman method for L1 regularized problems
   SIAM Journal on Imaging Sciences 2(2) 2009
  *Micchelli et al, Proximity algorithms for image models: denoising
   Inverse Problems 27(4) 2011
'''
# Benjamin Trmoulhac
# University College London
# b.tremoulheac@cs.ucl.ac.uk
# April 2012
def SB_ATV(g,mu):
    g = g.flatten()
    n = len(g)
    B,Bt,BtB = DiffOper(np.sqrt(n))
    b = np.zeros((2*n,1))
    d = b
    u = g
    err = 1
    k = 1
    tol = 1e-3
    lambda1 = 1 #Avoid using lambda because it is a keyword in Python
    while err > tol:
        print 'it. %g ',k
        up = u
        pdb.set_trace()
        u,_ = splinalg.cg(sp.eye(n)+BtB, g-lambda1*Bt*(b-d),tol=1e-5,maxiter=100)
        Bub = B*u+b
        d = max(np.abs(Bub)-mu/lambda1,0)*np.sign(Bub)
        b = Bub-d
        err = np.linalg.norm(up-u)/np.linalg.norm(u)
        print 'err=%g \n',err
        k = k+1
    print 'Stopped because norm(up-u)/norm(u) <= tol=%.1e\n',tol
    return u

def delete_row_lil(mat, i):
    if not isinstance(mat, scipy.sparse.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])

def delete_col_lil(mat, i):
    if not isinstance(mat, scipy.sparse.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.cols = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0],mat._shape[1]- 1)


def delete_row_csr(mat, i):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

def DiffOper(N):
    D = sp.spdiags(np.transpose(np.hstack((-np.ones((N,1)),np.ones((N,1))))),[0,1], N,N+1,"csr")
    #D[:,1] = []
    print 'shape before: ',D.shape
    D=D[:,1:]
    #D=np.delete(D,0,1) #delete the first column
    print 'shape afterward: ',D.shape
    #D=sp.csr_matrix(D)
    D[0,0] = 0
    #D[1,1] = 0
    B = sp.hstack([sp.kron(sp.eye(N),D),sp.kron(D,sp.eye(N))])

    Bt = B.transpose()
    BtB = Bt*B
    return B,Bt,BtB 
