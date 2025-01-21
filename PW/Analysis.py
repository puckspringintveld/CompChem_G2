# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:57:17 2025

@author: 20211382
"""

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
import numpy as np
from xgboost import XGBRegressor
from scipy.interpolate import RegularGridInterpolator
from pytessel import PyTessel
import matplotlib.pyplot as plt
from matplotlib import gridspec
from analytical import psi_plot #analytical solutions

def main():
    a_0=100
    gr=np.linspace(0,a_0-a_0/16,16)
    max_n=3
    grid=a_0/2
    h=(gr[1]-gr[0])
    
    #generate analytical solutions
    a_psis=[]
    for n in range(1, max_n + 1):  # Principal quantum number (n >= 1)
        for l in range(0, n):  # Azimuthal quantum number (0 <= l < n)
            for m in range(-l, l + 1):  # Magnetic quantum number (-l <= m <= l)
                psi, _, _, z = psi_plot(n, l, m, grid)
                a_psis.append(psi)
                
    #generate rmse and pw contour plots and iso surfaces
    rmse_list=[]
    rmse_rho_list=[]
    pw_psi_list=[]
    for i in range(len(a_psis)):
        pw_psi=np.load(f'PW_psis/{a_0} angstrum/Psi_{i}.npy')
        #normalize
        gamma = inner3d(pw_psi, pw_psi) * h ** 3
        pw_psi = pw_psi / np.sqrt(gamma)
        pw_psi_list.append(pw_psi)
        check_norm=np.sum(pw_psi**2*h ** 3)
        print(i,check_norm) 
        iso_surface(pw_psi.real, i,h,grid) #create iso surace
        plot(pw_psi.real,a_0) #plot contour plot
        rmse,rmse_r=RMSE(a_psis[i][:-1,:-1,:-1].real,pw_psi.real,h) #generate rmse
        rmse_list.append(rmse)
        rmse_rho_list.append(rmse_r)
        
        return rmse_rho_list

def inner3d(v1, v2):
    """
    Function:   calculates the inner product between two 3-dimensional vectors.
    Input:      v1:         first 3-dimensional vector.
                v2:         second 3-dimensional vector.
    Output:     sum3d:      result of the inner product in 3 dimensions.
    """
    new3d = np.vdot(v1.ravel(), v2.ravel())

    return np.real(new3d)

#RMSE
def RMSE(ana,pw,h):
    res=pw-ana
    res_rho=pw**2-ana**2 
    
    RMSE=np.sqrt(np.sum(res**2))
    RMSE_rho=np.sqrt(np.sum(res_rho**2))
    
    return RMSE,RMSE_rho

def iso_surface(pw_psi,i,h,grid):
    # Generate wavefunction (psi) and additional outputs based on the grid    
        # Compute the electron density (normalized wavefunction squared)
        
    density = pw_psi.real**2 * h**3

        # Flatten the density array for processing
    flat_density = density.flatten()

    # Sort density values in descending order
    sorted_density = np.sort(flat_density)[::-1]

    # Compute the cumulative sum
    cumulative_density = np.cumsum(sorted_density)
    print(f"\t Density: {cumulative_density[-1]}")

    # Find the isovalue corresponding to 95% electron density
    isovalue_index = np.searchsorted(cumulative_density, 0.90)
    isovalue = sorted_density[isovalue_index]
        
    # Create a unit cell based on the grid size
    unitcell = np.diag(np.ones(3) * grid/10)
        
    # Initialize PyTessel for isosurface generation
    pytessel = PyTessel()
    # Generate and save the positive isosurface
    vertices, normals, indices = pytessel.marching_cubes(pw_psi.flatten(), pw_psi.shape, unitcell.flatten(), isovalue)
    pytessel.write_ply(f'pw_pos_{i}.ply', vertices, normals, indices)

    #Generate and save the negative isosurface
    vertices, normals, indices = pytessel.marching_cubes(pw_psi.flatten(), pw_psi.shape, unitcell.flatten(), -isovalue)
    pytessel.write_ply(f'pw_neg_{i}.ply', vertices, normals, indices)
    
def plot(psi,a_0,res=16):
    """
    
    """
    pos=np.linspace(0,a_0-a_0/res,res)
    z_indices = np.linspace(0,  res, 9).astype(int)
    z_indices[-1] = res-1
    
    # Set up the figure and plot parameters
    fig = plt.figure(figsize=(13, 12))
    gs = gridspec.GridSpec(3, 3, wspace=0.3, hspace=0.3, right=0.85)
    extent = [0, a_0-a_0/res, 0, a_0-a_0/res]
    limit = np.max(np.abs(psi))

    # Generate contour plots for XY planes at selected Z slices
    for idx, z_index in enumerate(z_indices):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(
            psi[:, :, z_index].real,  # Use the real part of the wavefunction
            origin='lower',
            extent=extent,
            cmap='seismic',
            vmin=-limit,
            vmax=limit
        )
        ax.set_title(f"XY Plane at Z = {pos[z_index]:.2f} (a.u.)")
        ax.set_xlabel("x (a.u.)")
        ax.set_ylabel("y (a.u.)")

    # Add a colorbar to the figure
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Wavefunction Value")

    # Add a title and save the figure
    fig.suptitle(f"Energy: -0.03681 Ht", fontsize=16, y=0.92)
    fig.savefig('contour plots/3d5_pw', dpi=300, bbox_inches="tight")  # Ensure high-quality saving
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    rmse=main()
