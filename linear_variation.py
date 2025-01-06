import numpy as np
from pylebedev import PyLebedev
from pyqint import PyQInt, cgf

from gaussian_type_orbitals import GTO, H_GTO, normalization_constant
from analytical_integrals import kinetic_integral, nuclear_integral, overlap_integral

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

def energy_function(Alphas, Lebedevorder, Radialpoints, l, m, n, Z, method):
    
    if method == 0:
        H, S = Matrixes_Numerically(Alphas, Lebedevorder, Radialpoints, l, m, n, Z)
    elif method == 1:
        H, S = Matrixes_Analytically(Alphas, l, m, n, Z)
    elif method == 2:
        H, S = Matrixes_Analytically_Pyqint(Alphas, l, m, n, Z)

    energy, _ = Linear_Variation(H, S)
    
    return np.sum(energy)