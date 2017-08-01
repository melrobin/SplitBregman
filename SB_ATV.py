import numpy as np
import scipy.sparse as sp
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
    B,Bt,BtB = DiffOper(sqrt(n))
    b = np.zeros(2*n,1)
    d = b
    u = g
    err = 1
    k = 1
    tol = 1e-3
    lambda1 = 1 #Avoid using lambda because it is a keyword in Python
    while err > tol:
        print 'it. %g ',k
        up = u
        u,_ = sp.linalg.cg(sp.eye(n)+BtB, g-lambda1*Bt*(b-d),tol=1e-5,maxiter=100)
        Bub = B*u+b
        d = max(np.abs(Bub)-mu/lambda1,0)*np.sign(Bub)
        b = Bub-d
        err = np.linalg.norm(up-u)/np.linalg.norm(u)
        print 'err=%g \n',err
        k = k+1
    print 'Stopped because norm(up-u)/norm(u) <= tol=%.1e\n',tol
    return u

def DiffOper(N):
    D = sp.diags([-np.ones(N,1),np.ones(N,1)], [0 1], N,N+1)
    D[:,1] = []
    D[1,1] = 0
    B = [[ np.kron(sp.eye(N),D)],[ np.kron(D,sp.eye(N))]]
    Bt = np.tranpose(B)
    BtB = Bt*B
    return B,Bt,BtB 
