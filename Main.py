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
from split_bregman import SB_ATV,DiffOper
# Load an color image in grayscale
N = 512
n = N*N
f = cv2.imread('Lena512.png',0)
#g = f(:) + 0.09*max(f(:))*randn(n,1);
g=f.flatten('F')+0.09*np.amax(f.flatten())*np.random.randn(n)
mu = 20;
#pdb.set_trace()
g_denoise_atv = SB_ATV(g,mu);
#g_denoise_itv = SB_ITV(g,mu);

#print 'ATV Rel.Err = %g\n',np.linalg.norm(g_denoise_atv.flatten() - f.flatten()) / np.linalg.norm(f.flatten()) 
#print 'ITV Rel.Err = %g\n',np.linalg.norm(g_denoise_itv.flatten() - f.flatten()) / np.linalg.norm(f.flatten()) 
#
#figure; colormap gray;
#subplot(221); imagesc(f); axis image; title('Original');
#subplot(222); imagesc(reshape(g,N,N)); axis image; title('Noisy');
#subplot(223); imagesc(reshape(g_denoise_atv,N,N)); axis image; 
#title('Anisotropic TV denoising');
#subplot(224); imagesc(reshape(g_denoise_itv,N,N)); axis image; 
#title('Isotropic TV denoising');
