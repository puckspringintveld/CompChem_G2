# importing general python modules
import numpy as np
from scipy.special import factorial

def generate_lmn(final_sum):
    """
    Generates all possible combinations of (l, m, n) such that:
    - l, m, n >= 0
    - l + m + n = total_sum

    Args:
        final_sum (int): The target final sum for the numbers l, m, n.

    Returns:
        tuple of np.ndarray: Three numpy arrays containing the values of l, m, and n.
    """
    l_values, m_values, n_values = [], [], []
    # first loop is over the final sum
    for total_sum in range(final_sum + 1):
        # generating two more loops to create all the possible values of l, m, n >= 0
        for l in range(total_sum + 1):
            for m in range(total_sum - l + 1):
                n = total_sum - l - m
                
                # saving the values of l, m, and n
                l_values.append(l)
                m_values.append(m)
                n_values.append(n)
                
    return np.array(l_values), np.array(m_values), np.array(n_values)

def normalization_constant(Alphas, l, m, n):
    """
    Compute the normalization constant for a Gaussian-Type Orbital (GTO).

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

    Returns
    -------
    float
        The normalization constant for the GTO(s).
    """
    # Prefactor for normalization based on the Gaussian exponent
    prefactor = (2 * Alphas / np.pi) ** (3 / 4)

    # Angular momentum dependent factor
    angular_factor = ((8 * Alphas) ** (l + m + n) *
                      factorial(l) * factorial(m) * factorial(n) /
                      (factorial(2 * l) * factorial(2 * m) * factorial(2 * n))) ** 0.5

    # Return the combined normalization constant
    return prefactor * angular_factor


def GTO(x, y, z, alpha, l, m, n):
    """
    Gaussian-Type Orbital (GTO).
    Parameters:
        x, y, z : array-like
            Coordinates where the GTO is evaluated.
        alpha : float
            Gaussian exponent.
        l, m, n : int
            Angular momentum gto numbers.
    """
    # normalization constant
    N = normalization_constant(alpha, l, m, n) 
    
    return N * (x ** l) * (y ** m) * (z ** n) * np.exp(-alpha * (x**2 + y**2 + z**2))

def H_GTO(x, y, z, alpha, l, m, n, Z):
    """
    Hamiltonian acting on a Gaussian-Type Orbital (GTO).
    Parameters:
        x, y, z : array-like
            Coordinates where the GTO is evaluated.
        alpha : float
            Gaussian exponent.
        l, m, n : int
            Angular momentum gto numbers.
    """
    # normalization constant
    N = normalization_constant(alpha, l, m, n)
    
    # kinetic part of hamiltonian acting on the gaussian type orbital
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
    
    # potential part of hamiltonian acting on the gaussian type orbital
    Potential = - Z * N * (x ** l) * (y ** m) * (z ** n) * \
                  np.exp(-alpha * (x**2 + y**2 + z**2)) / (np.sqrt(x**2 + y**2 + z**2))

    return Potential + Kinetic