# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:33:38 2025

@author: 20211382
"""

import numpy as np
from pydft import MoleculeBuilder, DFT
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy as sp

# perform DFT calculation on the CO molecule
co = MoleculeBuilder().from_name("CO")
print(co)
dft = DFT(co, basis='sto3g')
en = dft.scf(1e-6)
result=dft.get_data()

P=result['P']
S=result['S']
p_C=result['C']
M_O=np.sum(P[:5,:]*S[:,:5].T)
M_C=np.sum(P[5:,:]*S[:,5:].T)
M_mu=M_O+M_C

# generate grid of points and calculate the electron density for these points
sz = 4      # size of the domain
npts = 100  # number of sampling points per cartesian direction


# produce meshgrid for the xz-plane
x = np.linspace(-sz/2,sz/2,npts)
zz, xx = np.meshgrid(x, x, indexing='ij')
gridpoints = np.zeros((zz.shape[0], xx.shape[1], 3))
gridpoints[:,:,0] = xx
gridpoints[:,:,2] = zz
gridpoints[:,:,1] = np.zeros_like(xx) # set y-values to 0
gridpoints = gridpoints.reshape((-1,3))

# calculate (logarithmic) scalar field and convert if back to an 2D array
density = dft.get_density_at_points(gridpoints)
density = np.log10(density.reshape((npts, npts)))

grid=x.copy()
y=[]
y_flux=[]
mins=[]
D=np.zeros((len(x),len(x)-1))
for i in range(len(x)):
    pos=np.where(density[:,i]==np.min(density[:,i]))
    mins.append(np.min(density[:,i]))
    y.append(x[pos]) 

for i in range(len(x)):
    derivative=density[:-1,i]-density[1:,i]/(x[1]-x[0])
    D[i,:]=derivative
    # derivative[density[:-1,i]>0]=10
    # p=np.where(np.round(derivative,2)==0)
    
    # # if np.shape(p[0])[0]==1:
    # #     while (derivative[p[0]]-derivative[p[0]-1])>0:
    # #         derivative[p]+=10
    # #         p=np.where(np.round(derivative,1)==0)
    # # else:
    # #     p_new=0
    # #     for j in range(np.shape(p[0])[0]):
    # #         if (derivative[p[0][j]]-derivative[p[0][j]-1])<0:
    # #             p_new=p[0][j]
    # #         else:
    # #             derivative[p[0][j]]+=10
    # #             p_new=np.where(np.round(derivative,1)==0)
    # #     p=p_new
                
    # #p=np.where(derivative**2==np.min(derivative**2))
    # y_flux.append(x[p])
    

# build contour plot
fig, ax = plt.subplots(1,1, dpi=144, figsize=(4,4))
im = ax.contourf(x, x, density, levels=np.linspace(-3,3,13, endpoint=True))
ax.contour(x, x, density, colors='black', levels=np.linspace(-3,3,13, endpoint=True))
ax.plot(x,y,c='r')
ax.plot(x,y_flux,c='y')
ax.set_aspect('equal', 'box')
ax.set_xlabel('x-coordinate [a.u.]')
ax.set_ylabel('z-coordinate [a.u.]')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_title('Electron density')

plt.plot(y,mins, '.')
plt.show()