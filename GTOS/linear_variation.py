# importing general python modules
import numpy as np
from pylebedev import PyLebedev
from pyqint import PyQInt, cgf

# importing self created functions
from gaussian_type_orbitals import GTO, H_GTO, normalization_constant
from analytical_integrals import kinetic_integral, nuclear_integral, overlap_integral

def Linear_Variation(H, S):   
    """
    Solve the generalized eigenvalue problem for the linear variational method.

    Parameters
    ----------
    H : numpy.ndarray
        Hamiltonian matrix representing the system's energy terms.
    S : numpy.ndarray
        Overlap matrix representing the basis function overlap integrals.

    Returns
    -------
    ep : numpy.ndarray
        Eigenvalues of the transformed Hamiltonian (energy levels).
    C : numpy.ndarray
        Eigenvectors (coefficients) in the original basis set.
    """
    # Compute the eigenvalues and eigenvectors of the overlap matrix S
    e, v = np.linalg.eigh(S)
    
    # Construct the transformation matrix X to orthogonalize the basis
    X = v @ np.diag(1 / np.sqrt(e))
    
    # Transform the Hamiltonian into the orthogonalized basis
    Hp = X.transpose() @ H @ X
    
    # Solve the eigenvalue problem in the orthogonalized basis
    ep, Cp = np.linalg.eigh(Hp)
    
    # Transform the eigenvectors back to the original basis
    C = X @ Cp
    
    # Return the eigenvalues and transformed eigenvectors
    return ep, C

def Matrixes_Numerically(Alphas, Lebedevorder, Radialpoints, l, m, n, Z):
    """
    Compute the Hamiltonian and overlap matrices numerically using a Gaussian-Type Orbital (GTO) basis using Gauss-Chebyshev-Lebedev integration.

    Parameters
    ----------
    Alphas : list or numpy.ndarray
        Exponents of the Gaussian basis functions.
    Lebedevorder : int
        Order of the Lebedev quadrature for angular integration.
    Radialpoints : int
        Number of radial grid points for the Gauss-Chebychev quadrature.
    l : list or numpy.ndarray
        Angular momentum gto numbers in the x-direction for each basis function.
    m : list or numpy.ndarray
        Angular momentum gto numbers in the y-direction for each basis function.
    n : list or numpy.ndarray
        Angular momentum gto numbers in the z-direction for each basis function.
    Z : int
        Nuclear charge of the atom or molecule.

    Returns
    -------
    H : numpy.ndarray
        Hamiltonian matrix representing the system's energy terms.
    S : numpy.ndarray
        Overlap matrix representing the basis function overlap integrals.
    """
    rm = 0.35  # Scaling factor for radial coordinates

    # Build the Gauss-Chebychev radial grid
    z = np.arange(1, Radialpoints + 1)
    x = np.cos(np.pi / (Radialpoints + 1) * z)  # Cosine mapping for Chebyshev points
    r = rm * (1 + x) / (1 - x)  # Radial transformation
    wr = (np.pi / (Radialpoints + 1) * np.sin(np.pi / (Radialpoints + 1) * z)**2
          * 2.0 * rm / (np.sqrt(1 - x**2) * (1 - x)**2))  # Radial weights

    # Get Lebedev quadrature points and weights
    leblib = PyLebedev()
    p, wl = leblib.get_points_and_weights(Lebedevorder)

    # Construct the full 3D grid by combining radial and angular grids
    gridpts = np.outer(r, p).reshape((-1, 3))  # Combine radial and angular points
    gridw = np.outer(wr * r**2, wl).flatten()  # Combine radial and angular weights

    # Separate x, y, z coordinates from the grid
    x = gridpts[:, 0]
    y = gridpts[:, 1]
    z = gridpts[:, 2]

    # Prevent division by zero in the GTO computation
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

    # Evaluate the Gaussian-Type Orbitals (GTOs) on the grid
    Psis = GTO(x, y, z, Alphas, l, m, n)

    # Evaluate the Hamiltonian applied to the GTOs
    H_Psis = H_GTO(x, y, z, Alphas, l, m, n, Z)

    # Compute the Hamiltonian matrix via numerical integration
    H = 4 * np.pi * np.einsum('li,lj,l->ij', Psis, H_Psis, gridw)

    # Compute the overlap matrix via numerical integration
    S = 4 * np.pi * np.einsum('li,lj,l->ij', Psis, Psis, gridw)

    # Return the Hamiltonian and overlap matrices
    return H, S


def Matrixes_Analytically_Pyqint(Alphas, l, m, n, Z):
    """
    Compute the Hamiltonian and overlap matrices analytically using PyQInt.

    Parameters
    ----------
    Alphas : list or numpy.ndarray
        Exponents of the Gaussian basis functions.
    l : list or numpy.ndarray
        Angular momentum gto numbers in the x-direction for each basis function.
    m : list or numpy.ndarray
        Angular momentum gto numbers in the y-direction for each basis function.
    n : list or numpy.ndarray
        Angular momentum gto numbers in the z-direction for each basis function.
    Z : float
        Nuclear charge of the atom or molecule.

    Returns
    -------
    H : numpy.ndarray
        Hamiltonian matrix representing the system's energy terms.
    S : numpy.ndarray
        Overlap matrix representing the basis function overlap integrals.
    """
    # Determine the number of Gaussian basis functions
    length = len(Alphas)
    
    # Initialize the Hamiltonian and overlap matrices with zeros
    H = np.zeros((length, length))
    S = np.zeros_like(H)
    
    # Initialize the PyQInt integrator for analytic integral calculations
    integrator = PyQInt()

    # Loop over all unique pairs of basis functions (including diagonal)
    for i in range(length):
        for j in range(i, length):  # Start at `i` to include diagonal elements
            # Create the Gaussian basis function for the `i`th function
            cgf_i = cgf([0.0, 0.0, 0.0])  # Centered at origin
            cgf_i.add_gto(1, Alphas[i], int(l[i]), int(m[i]), int(n[i]))
            
            # Create the Gaussian basis function for the `j`th function
            cgf_j = cgf([0.0, 0.0, 0.0])  # Centered at origin
            cgf_j.add_gto(1, Alphas[j], int(l[j]), int(m[j]), int(n[j]))

            # Compute the required integrals
            kinetic = integrator.kinetic(cgf_i, cgf_j)  # Kinetic energy integral
            nuclear = integrator.nuclear(cgf_i, cgf_j, [0.0, 0.0, 0.0], Z)  # Nuclear attraction integral
            overlap = integrator.overlap(cgf_i, cgf_j)  # Overlap integral

            # Fill the Hamiltonian and overlap matrices symmetrically
            H[i, j] = H[j, i] = kinetic + nuclear
            S[i, j] = S[j, i] = overlap

    # Return the computed Hamiltonian and overlap matrices
    return H, S


def Matrixes_Analytically(Alphas, l, m, n, Z):
    """
    Compute the Hamiltonian and overlap matrices analytically using self created analytical integrals.

    Parameters
    ----------
    Alphas : list or numpy.ndarray
        Exponents of the Gaussian basis functions.
    l : list or numpy.ndarray
        Angular momentum gto numbers in the x-direction for each basis function.
    m : list or numpy.ndarray
        Angular momentum gto numbers in the y-direction for each basis function.
    n : list or numpy.ndarray
        Angular momentum gto numbers in the z-direction for each basis function.
    Z : float
        Nuclear charge of the atom or molecule.

    Returns
    -------
    H : numpy.ndarray
        Hamiltonian matrix representing the system's energy terms.
    S : numpy.ndarray
        Overlap matrix representing the basis function overlap integrals.
    """
    # Determine the number of Gaussian basis functions
    length = len(Alphas)
    
    # Initialize the Hamiltonian and overlap matrices with zeros
    H = np.zeros((length, length))
    S = np.zeros_like(H)

    # Loop over all unique pairs of basis functions (including diagonal)
    for i in range(length):
        for j in range(i, length):  # Start at `i` to include diagonal elements

            # Compute normalization constants for the `i`th and `j`th functions
            Norm_i = normalization_constant(Alphas[i], l[i], m[i], n[i])
            Norm_j = normalization_constant(Alphas[j], l[j], m[j], n[j])

            # Calculate the required integrals
            kinetic = kinetic_integral(
                l[i], m[i], n[i], l[j], m[j], n[j], Alphas[i], Alphas[j], Norm_i, Norm_j
            )  # Kinetic energy integral
            nuclear = nuclear_integral(
                l[i], m[i], n[i], l[j], m[j], n[j], Z, Alphas[i], Alphas[j], Norm_i, Norm_j
            )  # Nuclear attraction integral
            overlap = overlap_integral(
                l[i], m[i], n[i], l[j], m[j], n[j], Alphas[i], Alphas[j], Norm_i, Norm_j
            )  # Overlap integral

            # Fill the Hamiltonian and overlap matrices symmetrically
            H[i, j] = H[j, i] = kinetic + nuclear
            S[i, j] = S[j, i] = overlap

    # Return the computed Hamiltonian and overlap matrices
    return H, S


def energy_function(Alphas, Lebedevorder, Radialpoints, l, m, n, Z, method):
    """
    Compute the total energy of the system using different matrix computation methods.

    Parameters
    ----------
    Alphas : list or numpy.ndarray
        Exponents of the Gaussian basis functions.
    Lebedevorder : int
        Order of the Lebedev quadrature for angular integration (used in numerical method).
    Radialpoints : int
        Number of radial grid points for the Gauss-Chebychev quadrature (used in numerical method).
    l : list or numpy.ndarray
        Angular momentum gto numbers in the x-direction for each basis function.
    m : list or numpy.ndarray
        Angular momentum gto numbers in the y-direction for each basis function.
    n : list or numpy.ndarray
        Angular momentum gto numbers in the z-direction for each basis function.
    Z : float
        Nuclear charge of the atom or molecule.
    method : int
        Method for computing matrices:
        - 0: Numerical Integration Via Gauss-Chebyshev-Lebedev.
        - 1: Analytical Solutions in Python.
        - 2: Analytical Solutions via Pyqint.

    Returns
    -------
    float
        The total energy of the system, computed as the sum of eigenvalues of the Hamiltonian.
    """
    # Compute the Hamiltonian and overlap matrices based on the chosen method
    if method == 0:
        # Numerical integration
        H, S = Matrixes_Numerically(Alphas, Lebedevorder, Radialpoints, l, m, n, Z)
    elif method == 1:
        # Analytical integration
        H, S = Matrixes_Analytically(Alphas, l, m, n, Z)
    elif method == 2:
        # Analytical integration using PyQInt
        H, S = Matrixes_Analytically_Pyqint(Alphas, l, m, n, Z)

    # Solve the generalized eigenvalue problem to get the energy levels
    energy, _ = Linear_Variation(H, S)

    # Return the total energy as the sum of eigenvalues
    return np.sum(energy)
