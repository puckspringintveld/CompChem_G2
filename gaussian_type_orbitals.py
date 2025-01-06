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

def normalization_constant(alpha, l, m, n):
    """Compute the normalization constant for a GTO."""
    prefactor = (2 * alpha / np.pi) ** (3 / 4)
    angular_factor = ((8 * alpha) ** (l + m + n) *
                      factorial(l) * factorial(m) * factorial(n) /
                      (factorial(2 * l) * factorial(2 * m) * factorial(2 * n))) ** 0.5
    
    return prefactor * angular_factor

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