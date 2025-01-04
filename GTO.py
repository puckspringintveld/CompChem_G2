# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:58:04 2024

@author: stuar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial 
from scipy.optimize import minimize
from pylebedev import PyLebedev
from pyqint import PyQInt, cgf
from matplotlib import gridspec
import time 


def main():
    Radialpoints = 2**7
    Lebedevorder = 17
    Z = 1 
    method = 2
    l = np.array([0,1,0,0,1,1,0,2,0,0,1,2,2,1,0,1,0,3,0,0,2,1,1,2,2,0,3,3,1,0,0,1,4,0,0,2,2,1,3,1,1,3,3,2,0,0,2,4,4,0,1,0,1,5,0,0,2,3,3,1,2,1,2,4,4,4,0,2,1,0,2,1,5,5,0,1,0,1,6,0,0])
    m = np.array([0,0,1,0,1,0,1,0,2,0,1,1,0,2,2,0,1,0,3,0,1,2,1,2,0,2,1,0,0,1,3,3,0,4,0,2,1,2,1,3,1,2,0,3,3,2,0,1,0,4,4,1,0,0,5,0,2,2,1,3,3,2,1,2,0,1,4,4,4,2,0,1,1,0,5,5,1,0,0,6,0])
    n = np.array([0,0,0,1,0,1,1,0,0,2,1,0,1,0,1,2,2,0,0,3,1,1,2,0,2,2,0,1,3,3,1,0,0,0,4,1,2,2,1,1,3,0,2,0,2,3,3,0,1,1,0,4,4,0,0,5,2,1,2,2,1,3,3,0,2,1,2,0,1,4,4,4,0,1,1,0,5,5,0,0,6])
    # l = np.array([0, 1, 0, 0])
    # m = np.array([0, 0, 1, 0])
    # n = np.array([0, 0, 0, 1])
    
    N = len(l)
    Beta = 1.2
    Alpha = 0.2
    X0 = np.array([Alpha * Beta ** -(i - 1)/i for i in range(1, N + 1)])    

    start_time = time.time()  # Record the start time
    
    result = minimize(energy_function, x0=X0, bounds=[(1e-8, 1e8)], args=(Lebedevorder, Radialpoints, l, m, n, Z, method))
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print(f"Optimized Alpha: {result.x}")

    optimized_result = result.x
    # Compute the final energy using the optimized Alpha
    H, S = Matrixes_Analytically_Pyqint(optimized_result, l, m, n, Z)
    energy, coefficients = Linear_Variation(H, S)
    print(f"Final Energy: {energy}")
    
    for i in range(N):
        plot(optimized_result, l, m, n, coefficients[:, i])
    
 
def plot(Alphas, l, m, n, coefficients):
    grid = auto_adjust_grid(Alphas, l, m, n, coefficients, 1e-2)
    
    psi, _, _, z =psi_plot(Alphas, l, m, n, coefficients, grid)

    # Compute z indices for slices
    z_indices = np.linspace(0, len(z) - 1, 9).astype(int)
    
    # Set up the figure
    fig = plt.figure(figsize=(13, 12))
    gs = gridspec.GridSpec(3, 3, wspace=0.3, hspace=0.3, right=0.85)
    extent = [-grid, grid, -grid, grid]
    limit = np.max(np.abs(psi))

    # Create subplots
    for idx, z_index in enumerate(z_indices):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(
            psi[:, :, z_index],
            origin='lower',
            extent=extent,
            cmap='PiYG',
            vmin=-limit,
            vmax=limit
        )
        ax.set_title(f"XY Plane at Z = {z[z_index]:.2f} (a.u.)")
        ax.set_xlabel("x (a.u.)")
        ax.set_ylabel("y (a.u.)")
    
    # Add a single colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Wavefunction Value")
    plt.show()

def psi_plot(Alphas, l, m, n, coefficients, grid):
    x = np.linspace(-grid, grid, 101)
    y = np.linspace(-grid, grid, 101)
    z = np.linspace(-grid, grid, 101)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')  # Proper axis alignment

    # Ensure proper reshaping for broadcasting
    Alphas = np.array(Alphas)[:, None, None, None]  # Shape (N, 1, 1, 1)
    l = np.array(l)[:, None, None, None]            # Shape (N, 1, 1, 1)
    m = np.array(m)[:, None, None, None]            # Shape (N, 1, 1, 1)
    n = np.array(n)[:, None, None, None]            # Shape (N, 1, 1, 1)
    coefficients = np.array(coefficients)[:, None, None, None]  # Shape (N, 1, 1, 1)

    # Add an axis for grid arrays to match the leading dimension (N)
    xx = xx[None, :, :, :]  # Shape (1, 101, 101, 101)
    yy = yy[None, :, :, :]  # Shape (1, 101, 101, 101)
    zz = zz[None, :, :, :]  # Shape (1, 101, 101, 101)

    # Numerically calculate psi
    psi_plot = np.sum(
        coefficients
        * normalization_constant(Alphas, l, m, n)  # Shape (N, 1, 1, 1)
        * (xx ** l) * (yy ** m) * (zz ** n)        # Shape (N, 101, 101, 101)
        * np.exp(-Alphas * (xx**2 + yy**2 + zz**2)),  # Shape (N, 101, 101, 101)
        axis=0  # Sum over Gaussian components
    )
    return psi_plot, x, y, z

def auto_adjust_grid(Alphas, l, m, n, coefficients, threshold_factor):
    """
    Automatically adjusts the grid size based on the wavefunction values.
    
    Parameters:
        psi_plot (ndarray): The computed wavefunction on the grid.
        x, y, z (ndarray): The grid points in each dimension.
        threshold_factor (float): Factor for determining significant values (default: 1e-3).
        padding (float): Padding around the bounding box (default: 2 units).
    
    Returns:
        Tuple: New ranges for x, y, z.
    """
    # Define the spatial grid
    psi, x, y, z = psi_plot(Alphas, l, m, n, coefficients, 100)
    
    threshold = threshold_factor * np.max(np.abs(psi))
    significant_mask = np.abs(psi) > threshold

    # Find the bounds of significant values
    x_indices, y_indices, z_indices = np.where(significant_mask)
    x_min, x_max = x[x_indices.min()], x[x_indices.max()]
    y_min, y_max = y[y_indices.min()], y[y_indices.max()]
    z_min, z_max = z[z_indices.min()], z[z_indices.max()]

    return max(abs(np.array([x_min, x_max, y_min, y_max, z_min, z_max])))
    
def GTO(x, y, z, alpha, l, m, n):
    """
    Gaussian-Type Orbital (GTO) implementation with proper normalization.
    Parameters:
        x, y, z : array-like
            Coordinates where the GTO is evaluated.
        alpha : float
            Gaussian exponent.
        l, m, n : int
            Angular momentum quantum numbers.
    """

    N = normalization_constant(alpha, l, m, n)
    
    return N * (x ** l) * (y ** m) * (z ** n) * np.exp(-alpha * (x**2 + y**2 + z**2))

def H_GTO(x, y, z, alpha, l, m, n, Z):
    """
    Gaussian-Type Orbital (GTO) implementation with proper normalization.
    Parameters:
        x, y, z : array-like
            Coordinates where the GTO is evaluated.
        alpha : float
            Gaussian exponent.
        l, m, n : int
            Angular momentum quantum numbers.
    """

    N = normalization_constant(alpha, l, m, n)
    
    Kinetic = - (np.exp(-alpha * (x**2 + y**2 + z**2)) * 
                    N * x**(-2 + l) * y**(-2 + m) * z**(-2 + n) * 
                    (
                        (-1 + n) * n * x**2 * y**2 +
                        (
                            (-1 + l) * l * y**2 +
                            x**2 * (
                                (-1 + m) * m +
                                2 * alpha * (-3 - 2 * l - 2 * m - 2 * n + 2 * alpha * x**2) * y**2 +
                                4 * alpha**2 * y**4
                            )
                        ) * z**2 +
                        4 * alpha**2 * x**2 * y**2 * z**4
                    )
                ) / 2
    
    Potential = - Z * N * (x ** l) * (y ** m) * (z ** n) * \
                  np.exp(-alpha * (x**2 + y**2 + z**2)) / (np.sqrt(x**2 + y**2 + z**2))

    return Potential + Kinetic

def Linear_Variation(H, S):   
            
    e, v = np.linalg.eigh(S)
    X = v @ np.diag(1 / np.sqrt(e))
    
    Hp = X.transpose() @ H @ X
    ep, Cp = np.linalg.eigh(Hp)
    C = X @ Cp
    
    return ep, C

def Matrixes_Numerically(Alphas, Lebedevorder, Radialpoints, l, m, n, Z):
    rm = 0.35

    # build the Gauss-Chebychev grid following the recipe in the exercise
    z = np.arange(1, Radialpoints+1)
    x = np.cos(np.pi / (Radialpoints+1) * z)
    r = rm * (1 + x) / (1 - x)
    wr = np.pi / (Radialpoints+1) * np.sin(np.pi / (Radialpoints+1) * z)**2 * 2.0 * rm \
        / (np.sqrt(1 - x**2) * (1 - x)**2)

    # get Lebedev points
    leblib = PyLebedev()
    p,wl = leblib.get_points_and_weights(Lebedevorder)
    
    # construct full grid
    gridpts = np.outer(r, p).reshape((-1,3))
    gridw = np.outer(wr * r**2, wl).flatten()
    
    x = gridpts[:,0]
    y = gridpts[:,1]
    z = gridpts[:,2]
    
    epsilon = 1e-10
    x = np.where(x == 0, epsilon, x)
    y = np.where(y == 0, epsilon, y)
    z = np.where(z == 0, epsilon, z)
    
    # Convert inputs to NumPy arrays and adjust shapes for broadcasting
    x = np.asarray(x)[:, None]  # Shape (len(x), 1)
    y = np.asarray(y)[:, None]  # Shape (len(y), 1)
    z = np.asarray(z)[:, None]  # Shape (len(z), 1)
    Alphas = np.asarray(Alphas)[None, :]  # Shape (1, len(alpha))
    l = np.asarray(l)[None, :]  # Shape (1, len(alpha))
    m = np.asarray(m)[None, :]  # Shape (1, len(alpha))
    n = np.asarray(n)[None, :]  # Shape (1, len(alpha))


    Psis = GTO(x, y, z, Alphas, l, m, n)
    H_Psis = H_GTO(x, y, z, Alphas, l, m, n, Z)
    
    H = 4 * np.pi * np.einsum('li,lj,l->ij', Psis, H_Psis, gridw)
    S = 4 * np.pi * np.einsum('li,lj,l->ij', Psis, Psis, gridw)
    
    return H, S

def Matrixes_Analytically_Pyqint(Alphas, l, m, n, Z):
    length = len(Alphas)
    
    H = np.zeros((length, length))
    
    S = np.zeros_like(H)
    
    integrator = PyQInt()

    for i in range(length):
        for j in range(i, length):  # Include diagonal elements by starting at `i`
            # Create Gaussian functions for both indices
            cgf_i = cgf([0.0, 0.0, 0.0])
            cgf_i.add_gto(1, Alphas[i], int(l[i]), int(m[i]), int(n[i]))
            
            cgf_j = cgf([0.0, 0.0, 0.0])
            cgf_j.add_gto(1, Alphas[j], int(l[j]), int(m[j]), int(n[j]))

            # Calculate the required integrals
            kinetic = integrator.kinetic(cgf_i, cgf_j)
            nuclear = integrator.nuclear(cgf_i, cgf_j, [0.0, 0.0, 0.0], Z)
            overlap = integrator.overlap(cgf_i, cgf_j)

            # Fill the Hamiltonian and overlap matrices
            H[i, j] = H[j, i] = kinetic + nuclear
            S[i, j] = S[j, i] = overlap

    return H, S

def Matrixes_Analytically(Alphas, l, m, n, Z):
    length = len(Alphas)
    
    H = np.zeros((length, length))
    
    S = np.zeros_like(H)
    

    for i in range(length):
        for j in range(i, length):  # Include diagonal elements by starting at `i`

            Norm_i = normalization_constant(Alphas[i], l[i], m[i], n[i])
            Norm_j = normalization_constant(Alphas[j], l[j], m[j], n[j])

            # Calculate the required integrals
            kinetic = kinetic_integral(l[i], m[i], n[i], l[j], m[j], n[j], Alphas[i], Alphas[j], Norm_i, Norm_j)
            nuclear = nuclear_integral(l[i], m[i], n[i], l[j], m[j], n[j], Z, Alphas[i], Alphas[j], Norm_i, Norm_j)
            overlap = overlap_integral(l[i], m[i], n[i], l[j], m[j], n[j], Alphas[i], Alphas[j], Norm_i, Norm_j)

            # Fill the Hamiltonian and overlap matrices
            H[i, j] = H[j, i] = kinetic + nuclear
            S[i, j] = S[j, i] = overlap

    return H, S

def normalization_constant(alpha, l, m, n):
    """Compute the normalization constant for a GTO."""
    prefactor = (2 * alpha / np.pi) ** (3 / 4)
    angular_factor = ((8 * alpha) ** (l + m + n) *
                      factorial(l) * factorial(m) * factorial(n) /
                      (factorial(2 * l) * factorial(2 * m) * factorial(2 * n))) ** 0.5
    
    return prefactor * angular_factor


def energy_function(Alphas, Lebedevorder, Radialpoints, l, m, n, Z, method):
    
    if method == 0:
        H, S = Matrixes_Numerically(Alphas, Lebedevorder, Radialpoints, l, m, n, Z)
    elif method == 1:
        H, S = Matrixes_Analytically(Alphas, l, m, n, Z)
    elif method == 2:
        H, S = Matrixes_Analytically_Pyqint(Alphas, l, m, n, Z)

    energy, _ = Linear_Variation(H, S)
    
    return np.sum(energy)

def s(q1, q2, alpha1, alpha2):

    gamma = alpha1 + alpha2

    i1 = np.arange(0, int(q1 // 2) + 1)
    i2 = np.arange(0, int(q2 // 2) + 1)

    i1_mesh, i2_mesh = np.meshgrid(i1, i2, indexing="ij")
    i1, i2 = i1_mesh.flatten(), i2_mesh.flatten()
    
    omega = q1 + q2 - 2 * (i1 + i2) 
    
    omega_divided_plus_1 = omega // 2 + 1

    ranges = np.tile(np.arange(omega_divided_plus_1.max()), (omega_divided_plus_1.size, 1))  
    o = ranges[ranges < (omega_divided_plus_1)[:, None]] 

    i1 = np.repeat(i1, omega_divided_plus_1)
    i2 = np.repeat(i2, omega_divided_plus_1)
    omega = np.repeat(omega, omega_divided_plus_1)
        
    return (-1) ** q1 * factorial(q1) * factorial(q2) / gamma ** (q1 + q2) \
            * np.sum((-1) ** o
                     * factorial(omega)
                     * alpha1 ** (q2 - 2 * i2 - i1 - o)
                     * alpha2 ** (q1 - 2 * i1 - i2 - o)
                     / (4 ** (i1 + i2 + o))
                     / (factorial(i1) * factorial(i2) * factorial(o))
                     / (factorial(q1 - 2 * i1) * factorial(q2 - 2 * i2) * factorial(omega - 2 * o))
                     * gamma ** (2 * (i1 + i2) + o)
                     * 0 ** (omega - 2*o))

def unnorm_overlap(l1, m1, n1, l2, m2, n2, alpha1, alpha2):
    
    return (np.pi / (alpha1 + alpha2)) ** 1.5 * s(l1, l2, alpha1, alpha2) * s(m1, m2, alpha1, alpha2) * s(n1, n2, alpha1, alpha2)

def overlap_integral(l1, m1, n1, l2, m2, n2, alpha1, alpha2, N1, N2):
    
    return N1 * N2 * unnorm_overlap(l1, m1, n1, l2, m2, n2, alpha1, alpha2)

def kinetic_integral(l1, m1, n1, l2, m2, n2, alpha1, alpha2, N1, N2):
    
    """Compute the kinetic energy integral."""
    # Core overlap term
    term0 = alpha2 * (2 * (l2 + m2 + n2) + 3) * unnorm_overlap(l1, m1, n1, l2, m2, n2, alpha1, alpha2)

    # Incremented overlaps
    term1 = -2.0 * alpha2 ** 2 * (
        unnorm_overlap(l1, m1, n1, l2 + 2, m2, n2, alpha1, alpha2)
        + unnorm_overlap(l1, m1, n1, l2, m2 + 2, n2, alpha1, alpha2)
        + unnorm_overlap(l1, m1, n1, l2, m2, n2 + 2, alpha1, alpha2)
    )

    # Decremented overlaps
    term2 = 0
    if l2 - 2 >= 0:
        term2 += -0.5 * l2 * (l2 - 1) * unnorm_overlap(l1, m1, n1, l2 - 2, m2, n2, alpha1, alpha2)
        
    if m2 - 2 >= 0:
        term2+= -0.5 * m2 * (m2 - 1) * unnorm_overlap(l1, m1, n1, l2, m2 - 2, n2, alpha1, alpha2)
         
    if n2 - 2 >= 0:
        term2 += -0.5 * n2 * (n2 - 1) * unnorm_overlap(l1, m1, n1, l2, m2, n2 - 2, alpha1, alpha2)

    # Total kinetic energy
    return (term0 + term1 + term2) * N1 * N2

def binomial(a, b):
    
    return factorial(a) / (factorial(b) * factorial(a - b))

def binomial_prefactor(s, ia, ib):
    # Maximum possible t values across all s
    max_t = np.minimum(ib, s.max()) + 1
    
    # Create a 2D grid for all possible t values
    t = np.arange(max_t)
    t = np.tile(t, (len(s), 1))
    
    # Mask to identify valid t values for each s
    valid_t_mask = (t >= np.maximum(0, s[:, None] - ia)) & (t <= np.minimum(ib, s)[:, None])
    
    # Compute terms for all valid t values
    terms = np.zeros_like(t, dtype=float)
    valid_indices = valid_t_mask.nonzero()
    valid_s = valid_indices[0]
    valid_t = t[valid_t_mask]
    
    terms[valid_t_mask] = (
        binomial(ia, s[valid_s] - valid_t) *
        binomial(ib, valid_t) *
        0 ** (ia - s[valid_s] + valid_t) *
        0 ** (ib - valid_t)
    )
    
    return np.sum(terms, axis=1)

def A_array(l1, l2, gamma):
    imax = l1 + l2 + 1
    arrA = np.zeros(imax)
    
    i = np.arange(0, imax)
        
    i_divided_plus_1 = i // 2 + 1
    
    ranges = np.tile(np.arange(i_divided_plus_1.max()), (i_divided_plus_1.size, 1))  
    r = ranges[ranges < (i_divided_plus_1)[:, None]] 
    
    i = np.repeat(i, i_divided_plus_1)

    i_minus_2r_divided_plus_1 = (i - 2*r) // 2 + 1
    ranges = np.tile(np.arange(i_minus_2r_divided_plus_1.max()), (i_minus_2r_divided_plus_1.size, 1))  
    
    u = ranges[ranges < (i_minus_2r_divided_plus_1)[:, None]] 
    
    r = np.repeat(r, i_minus_2r_divided_plus_1)
    i = np.repeat(i, i_minus_2r_divided_plus_1)
    
    iI = i - 2 * r - u
        
    np.add.at(arrA, iI, ((-1) ** i
                        * binomial_prefactor(i, l1, l2) # binomial prefactor
                        * (-1) ** u
                        * factorial(i)
                        * 0 ** (i - 2 * r - 2 * u)
                        * (0.25 / gamma) ** (r + u)
                        / factorial(r)
                        / factorial(u)
                        / factorial(i - 2 * r - 2 * u)))

    return arrA
    
def nuclear_integral(l1, m1, n1, l2, m2, n2, Z, alpha1, alpha2, N1, N2):

    gamma = alpha1 + alpha2

    Ax = A_array(l1, l2, gamma)
    Ay = A_array(m1, m2, gamma) 
    Az = A_array(n1, n2, gamma) 
    
    # Create indices for Ax, Ay, and Az
    i, j, k = np.array(np.meshgrid(np.arange(l1 + l2 + 1),
                                   np.arange(m1 + m2 + 1),
                                   np.arange(n1 + n2 + 1),
                                   indexing='ij')).reshape(3, -1)

    boys = np.sum(1 / (2 * (i + j + k) + 1) * Ax[i] * Ay[j] * Az[k])
    
    return - Z * N1 * N2 * np.pi / gamma * 2 * boys

if __name__ == "__main__":
    main()