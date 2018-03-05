'''
This file demonstrates the Split Bregman method for Total Variation denoising

   SB_ATV.m  Split Bregman Anisotropic Total Variation Denoising
   SB_ITV.m  Split Bregman Isotropic Total Variation Denoising
'''
# Benjamin Trmoulhac
# University College London
# b.tremoulheac@cs.ucl.ac.uk
# April 2012
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
from split_bregman import SB_ATV,DiffOper,SB_ITV
# Load an color image in grayscale
N = 512
n = N*N
f = cv2.imread('Lena512.png',0)
g=f.flatten('F')+0.09*np.amax(f.flatten('F'))*np.random.randn(n)
mu = 20
g_denoise_atv = SB_ATV(g,mu)
g_denoise_itv = SB_ITV(g,mu)

print 'ATV Rel.Err = %g\n'%(np.linalg.norm(g_denoise_atv.flatten('F') - f.flatten('F')) / np.linalg.norm(f.flatten('F')))
print 'ITV Rel.Err = %g\n'%(np.linalg.norm(g_denoise_itv.flatten('F') - f.flatten('F')) / np.linalg.norm(f.flatten('F')))

fig=plt.figure()
ax=fig.add_subplot(2,2,1)
ax.imshow(f,cmap='gray')
ax.set_title('Original');
ax=fig.add_subplot(2,2,2)
ax.imshow(np.reshape(g,(N,N),order='F'),cmap='gray')
ax.set_title('Noisy')
ax=fig.add_subplot(2,2,3) 
plt.imshow(np.reshape(g_denoise_atv,(N,N),order='F'),cmap='gray')
ax.set_title('Anisotropic TV denoising')
ax=fig.add_subplot(2,2,4) 
plt.imshow(np.reshape(g_denoise_itv,(N,N),order='F'),cmap='gray')
ax.set_title('Isotropic TV denoising');
plt.show()
