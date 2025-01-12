# importing general python modules
import numpy as np
from sympy import Ynm
from sympy import Symbol, lambdify
from scipy.special import factorial

def generate_quantum_numbers(n_max):
    """
    Generate quantum numbers (n, l, m) up to a maximum principal quantum number.

    Parameters:
        n_max (int): The maximum value of the principal quantum number (n).

    Returns:
        tuple: Three NumPy arrays containing:
            - n_list (numpy.ndarray): Array of principal quantum numbers (n).
            - l_list (numpy.ndarray): Array of azimuthal quantum numbers (l).
            - m_list (numpy.ndarray): Array of magnetic quantum numbers (m).
    """
    n_list = []  # List to store principal quantum numbers (n).
    l_list = []  # List to store azimuthal quantum numbers (l).
    m_list = []  # List to store magnetic quantum numbers (m).

    # Iterate over all principal quantum numbers from 1 to n_max.
    for n in range(1, n_max + 1):
        # Iterate over all possible azimuthal quantum numbers for the given n.
        for l in range(n):
            # Iterate over all possible magnetic quantum numbers for the given l.
            for m in range(-l, l + 1):
                n_list.append(n)  # Append the current n to n_list.
                l_list.append(l)  # Append the current l to l_list.
                m_list.append(m)  # Append the current m to m_list.

    # Convert lists to NumPy arrays and return them.
    return np.array(n_list), np.array(l_list), np.array(m_list)


def spherical_harmonic(l, m, Theta, Phi):
    """
    Compute the real part of the tesseral spherical harmonic function for given quantum numbers.

    Parameters:
        l (int): Azimuthal quantum number (non-negative integer).
        m (int): Magnetic quantum number (integer, -l ≤ m ≤ l).
        Theta (float or array-like): Polar angle in radians (0 ≤ Theta ≤ π).
        Phi (float or array-like): Azimuthal angle in radians (0 ≤ Phi < 2π).

    Returns:
        numpy.ndarray or float: Real part of the tesseral spherical harmonic function .

    """
    # Define symbolic variables for the spherical harmonics
    phi = Symbol("phi", real=True)  # Symbolic variable for azimuthal angle
    theta = Symbol("theta", real=True)  # Symbolic variable for polar angle

    # Compute the spherical harmonic Y_l^m and its negative counterpart Y_l^-m
    ynm = Ynm(l, m, theta, phi).expand(func=True)  # Expand Y_l^m into a detailed form
    ynm_negm = Ynm(l, -m, theta, phi).expand(func=True)  # Expand Y_l^-m similarly

    # Create numerical functions for Y_l^m and Y_l^-m
    ynm_func = lambdify((theta, phi), ynm, modules="numpy")
    ynm_func_negm = lambdify((theta, phi), ynm_negm, modules="numpy")

    # Apply tesseral harmonic transformation based on the value of m
    if m < 0:
        # For negative m, compute the imaginary component and combine it to form the real-valued function
        yl_m = ynm_func(Theta, Phi)
        yl_neg_m = ynm_func_negm(Theta, Phi)
        ylm = 1j * (yl_m - (-1.0)**m * yl_neg_m) / np.sqrt(2)
    elif m > 0:
        # For positive m, compute the real component using a symmetric combination
        yl_m = ynm_func(Theta, Phi)
        yl_neg_m = ynm_func_negm(Theta, Phi)
        ylm = (yl_neg_m + (-1.0)**m * yl_m) / np.sqrt(2)
    else:
        # For m = 0, use Y_l^0 directly as it is purely real
        ylm = ynm_func(Theta, Phi)

    # Return the real part of the tesseral spherical harmonic due to numerical very small 1e-16 imaginary part needs to be removed
    return np.real(ylm)

def normalization_constant_STO(zeta, n):
    """
    Calculate the normalization constant for a Slater-Type Orbital (STO).

    Parameters:
        zeta (float): Orbital exponent (related to the effective nuclear charge).
        n (int): Principal quantum number of the STO.

    Returns:
        float: The normalization constant for the STO.
    """
    # Compute the normalization constant using the given formula
    return (2 * zeta)**n * np.sqrt((2 * zeta) / factorial(2 * n))


def STO(n, l, m, zeta, R, Theta, Phi):
    """
    Compute the Slater-Type Orbital (STO) wavefunction.

    Parameters:
        n (int): Principal quantum number.
        l (int): Azimuthal quantum number (non-negative integer).
        m (int): Magnetic quantum number (integer, -l ≤ m ≤ l).
        zeta (float): Orbital exponent (related to the effective nuclear charge).
        R (float or array-like): Radial distance from the nucleus.
        Theta (float or array-like): Polar angle in radians (0 ≤ Theta ≤ π).
        Phi (float or array-like): Azimuthal angle in radians (0 ≤ Phi < 2π).

    Returns:
        numpy.ndarray or float: The value of the STO wavefunction at the given (R, Theta, Phi).
    """
    # Compute the radial part of the STO wavefunction
    RadialPart = normalization_constant_STO(zeta, n) * R ** (n - 1) * np.exp(-zeta * R)
    
    # Compute the angular part of the STO wavefunction using spherical harmonics
    AngularPart = spherical_harmonic(l, m, Theta, Phi)
    
    # Combine the radial and angular parts to compute the STO wavefunction
    return RadialPart * AngularPart

def H_STO(n, l, zeta, R, Z, Psi):
    """
    Compute the total energy of a Slater-Type Orbital (STO).
    
    Parameters:
        n: Principal quantum number
        l: Angular momentum quantum number
        m: Magnetic quantum number
        zeta: Slater orbital exponent
        R: Radial distance
        Theta: Azimuthal angle
        Phi: Polar angle
        Z: Nuclear charge
        Psi: Wavefunction
    
    Returns:
        Total energy (Kinetic + Potential)
    """
    # Radial Laplacian contribution
    Kinetic = -0.5 * (zeta**2 - 2 * n * zeta / R + (n * (n - 1) - l*(l + 1) )/ R**2) * Psi

    # Potential energy term
    Potential = -Z * Psi / R

    # Combine terms
    return Kinetic + Potential
