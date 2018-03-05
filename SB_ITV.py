% Split Bregman Isotropic Total Variation Denoising
%
%   u = arg min_u 1/2||u-g||_2^2 + mu*ITV(u)
%
% Refs:
%  *Goldstein and Osher, The split Bregman method for L1 regularized problems
%   SIAM Journal on Imaging Sciences 2(2) 2009
%  *Micchelli et al, Proximity algorithms for image models: denoising
%   Inverse Problems 27(4) 2011
%
% Benjamin Trémoulhéac
% University College London
% b.tremoulheac@cs.ucl.ac.uk
% April 2012
import numpy as np, scipy.sparse as sp
def SB_ITV(g, mu):
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
     while err > tol
        print 'it. %g ', k
        up = u
        u,_=sp.linalg.cg(sp.eye(n)+BtB,g-lambda1*Bt*(b-d),tol=1e-5, maxiter=100)
        Bub = B * u + b
        s = np.sqrt(Bub[:n]**2 + Bub[n:]**2)
        #d = max(np.abs(Bub) - mu / lambda1, 0) * np.sign(Bub)
        d = np.vstack(max(s-mu/lambda,0).*Bub[:n]/s,
                max(s-mu/lambda,0).*Bub[n:]/s)
        b = Bub - d
        err = np.linalg.norm(up - u) / np.linalg.norm(u)
        print 'err=%g \n'% err
        k = k + 1

end
fprintf('Stopped because norm(up-u)/norm(u) <= tol=%.1e\n',tol);
end

#function [B Bt BtB] = DiffOper(N)
#D = spdiags([-ones(N,1) ones(N,1)], [0 1], N,N+1);
#D(:,1) = [];
#D(1,1) = 0;
#B = [ kron(speye(N),D) ; kron(D,speye(N)) ];
#Bt = B';
#BtB = Bt*B;
#end
