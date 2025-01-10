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
charge_o=8-M_O
charge_c=6-M_C

mu_O=charge_o*2.26*1/2.541
mu_C=charge_c*2.26*1/2.542

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
    y.append(x[pos][0]) 

y_min=min(y)
min_place=np.where(np.array(y)==y_min)
y_new=np.array(y)
y_new[:min_place[0][0]]=y_min
y_new[min_place[0][-1]:]=y_min
    

# build contour plot
fig, ax = plt.subplots(1,1, dpi=144, figsize=(4,4))
im = ax.contourf(x, x, density, levels=np.linspace(-3,3,13, endpoint=True), cmap='PiYG')
ax.contour(x, x, density, colors='black', levels=np.linspace(-3,3,13, endpoint=True))
ax.plot(x,y_new,c='k', linewidth=2.5)
ax.set_aspect('equal', 'box')
ax.set_xlabel('x-coordinate [a.u.]')
ax.set_ylabel('z-coordinate [a.u.]')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_title('Electron density')

plt.plot(y,mins, '.')
plt.show()