# importing general python modules
import numpy as np
from scipy.special import factorial

def s(q1, q2, alpha1, alpha2):
    """
    Compute the integral of the overlap between two Gaussian Type Orbitals along one axis.

    Parameters
    ----------
    q1 : int
        gto number associated with the first Gaussian Type Orbital.
    q2 : int
        gto number associated with the second Gaussian Type Orbital.
    alpha1 : float
        Exponent of the first Gaussian Type Orbital.
    alpha2 : float
        Exponent of the second Gaussian Type Orbital.

    Returns
    -------
    float
        The value of the integral for the overlap along one axis.
    """
    # Sum of the Gaussian exponents
    gamma = alpha1 + alpha2

    # Generate indices for terms in the summation based on q1 and q2
    i1 = np.arange(0, int(q1 // 2) + 1)
    i2 = np.arange(0, int(q2 // 2) + 1)

    # Create a meshgrid of indices for all combinations of i1 and i2
    i1_mesh, i2_mesh = np.meshgrid(i1, i2, indexing="ij")
    i1, i2 = i1_mesh.flatten(), i2_mesh.flatten()
    
    # Calculate the power omega for the summation
    omega = q1 + q2 - 2 * (i1 + i2)
    
    # Calculate the range of terms for each omega
    omega_divided_plus_1 = omega // 2 + 1
    ranges = np.tile(np.arange(omega_divided_plus_1.max()), (omega_divided_plus_1.size, 1))
    o = ranges[ranges < (omega_divided_plus_1)[:, None]]

    # Repeat indices to match the shape of omega terms
    i1 = np.repeat(i1, omega_divided_plus_1)
    i2 = np.repeat(i2, omega_divided_plus_1)
    omega = np.repeat(omega, omega_divided_plus_1)
        
    # Compute the overlap integral using summation
    return (-1) ** q1 * factorial(q1) * factorial(q2) / gamma ** (q1 + q2) * np.sum(
        (-1) ** o
        * factorial(omega)
        * alpha1 ** (q2 - 2 * i2 - i1 - o)
        * alpha2 ** (q1 - 2 * i1 - i2 - o)
        / (4 ** (i1 + i2 + o))
        / (factorial(i1) * factorial(i2) * factorial(o))
        / (factorial(q1 - 2 * i1) * factorial(q2 - 2 * i2) * factorial(omega - 2 * o))
        * gamma ** (2 * (i1 + i2) + o)
        * 0 ** (omega - 2 * o)
    )

def unnorm_overlap(l1, m1, n1, l2, m2, n2, alpha1, alpha2):
    """
    Compute the unnormalized overlap integral between two Gaussian Type Orbitals (GTOs).

    Parameters
    ----------
    l1, m1, n1 : int
        GTO numbers associated with the angular momentum of the first Gaussian Type Orbital
        in the x, y, and z directions, respectively.
    l2, m2, n2 : int
        GTO numbers associated with the angular momentum of the second Gaussian Type Orbital
        in the x, y, and z directions, respectively.
    alpha1 : float
        Exponent of the first Gaussian Type Orbital.
    alpha2 : float
        Exponent of the second Gaussian Type Orbital.

    Returns
    -------
    float
        The value of the unnormalized overlap integral.
    """
    return (
        (np.pi / (alpha1 + alpha2)) ** 1.5
        * s(l1, l2, alpha1, alpha2)
        * s(m1, m2, alpha1, alpha2)
        * s(n1, n2, alpha1, alpha2)
    )

def overlap_integral(l1, m1, n1, l2, m2, n2, alpha1, alpha2, N1, N2):
    """
    Compute the normalized overlap integral between two Gaussian Type Orbitals (GTOs).

    Parameters
    ----------
    l1, m1, n1 : int
        GTO numbers associated with the angular momentum of the first Gaussian Type Orbital
        in the x, y, and z directions, respectively.
    l2, m2, n2 : int
        GTO numbers associated with the angular momentum of the second Gaussian Type Orbital
        in the x, y, and z directions, respectively.
    alpha1 : float
        Exponent of the first Gaussian Type Orbital.
    alpha2 : float
        Exponent of the second Gaussian Type Orbital.
    N1 : float
        Normalization constant of the first Gaussian Type Orbital.
    N2 : float
        Normalization constant of the second Gaussian Type Orbital.

    Returns
    -------
    float
        The value of the normalized overlap integral.
    """
    return N1 * N2 * unnorm_overlap(l1, m1, n1, l2, m2, n2, alpha1, alpha2)

def kinetic_integral(l1, m1, n1, l2, m2, n2, alpha1, alpha2, N1, N2):
    """
    Compute the kinetic energy integral between two Gaussian Type Orbitals (GTOs).

    Parameters
    ----------
    l1, m1, n1 : int
        GTO numbers associated with the angular momentum of the first Gaussian Type Orbital
        in the x, y, and z directions, respectively.
    l2, m2, n2 : int
        GTO numbers associated with the angular momentum of the second Gaussian Type Orbital
        in the x, y, and z directions, respectively.
    alpha1 : float
        Exponent of the first Gaussian Type Orbital.
    alpha2 : float
        Exponent of the second Gaussian Type Orbital.
    N1 : float
        Normalization constant of the first Gaussian Type Orbital.
    N2 : float
        Normalization constant of the second Gaussian Type Orbital.

    Returns
    -------
    float
        The value of the kinetic energy integral.
    """
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
        term2 += -0.5 * m2 * (m2 - 1) * unnorm_overlap(l1, m1, n1, l2, m2 - 2, n2, alpha1, alpha2)
    if n2 - 2 >= 0:
        term2 += -0.5 * n2 * (n2 - 1) * unnorm_overlap(l1, m1, n1, l2, m2, n2 - 2, alpha1, alpha2)

    # Total kinetic energy
    return (term0 + term1 + term2) * N1 * N2

def binomial(a, b):
    """
    Compute the binomial coefficient, also known as "n choose k".

    Parameters
    ----------
    a : int
        The total number of items (n) in the set.
    b : int
        The number of items to choose (k) from the set.

    Returns
    -------
    float
        The binomial coefficient, computed as a! / (b! * (a - b)!).
    """
    return factorial(a) / (factorial(b) * factorial(a - b))

def binomial_prefactor(s, ia, ib):
    """
    Compute the binomial prefactor for given parameters.

    Parameters
    ----------
    s : numpy.ndarray
        Array of values representing the total number of items chosen.
    ia : int
        Number of items in the first subset.
    ib : int
        Number of items in the second subset.

    Returns
    -------
    numpy.ndarray
        The computed binomial prefactor for each value in `s`.
    """
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
        binomial(ia, s[valid_s] - valid_t) *  # Binomial coefficient for subset ia
        binomial(ib, valid_t) *              # Binomial coefficient for subset ib
        0 ** (ia - s[valid_s] + valid_t) *   
        0 ** (ib - valid_t)                
    )

    # Sum terms for each value in s
    return np.sum(terms, axis=1)

def A_array(l1, l2, gamma):
    """
    Compute the A array coefficients for two Gaussian Type Orbitals (GTOs).

    Parameters
    ----------
    l1 : int
        GTO number for the angular momentum in the first GTO along a specific axis.
    l2 : int
        GTO number for the angular momentum in the second GTO along the same axis.
    gamma : float
        Combined Gaussian exponent, defined as alpha1 + alpha2.

    Returns
    -------
    numpy.ndarray
        Array containing the computed A coefficients.
    """
    # Maximum value of i based on the angular momentum quantum numbers
    imax = l1 + l2 + 1
    arrA = np.zeros(imax)  # Initialize the A array with zeros
    
    # Generate index values for i
    i = np.arange(0, imax)
    
    # Compute the ranges for summation over r
    i_divided_plus_1 = i // 2 + 1
    ranges = np.tile(np.arange(i_divided_plus_1.max()), (i_divided_plus_1.size, 1))
    r = ranges[ranges < (i_divided_plus_1)[:, None]]
    
    # Expand i to account for repeated values
    i = np.repeat(i, i_divided_plus_1)

    # Compute the ranges for summation over u
    i_minus_2r_divided_plus_1 = (i - 2 * r) // 2 + 1
    ranges = np.tile(np.arange(i_minus_2r_divided_plus_1.max()), (i_minus_2r_divided_plus_1.size, 1))
    u = ranges[ranges < (i_minus_2r_divided_plus_1)[:, None]]

    # Expand r and i to match the size of u
    r = np.repeat(r, i_minus_2r_divided_plus_1)
    i = np.repeat(i, i_minus_2r_divided_plus_1)
    
    # Compute the remaining indices for the A array
    iI = i - 2 * r - u
    
    # Update the A array using the computed terms
    np.add.at(
        arrA, 
        iI, 
        ((-1) ** i
         * binomial_prefactor(i, l1, l2)  # Binomial prefactor
         * (-1) ** u
         * factorial(i)
         * 0 ** (i - 2 * r - 2 * u)  
         * (0.25 / gamma) ** (r + u)
         / factorial(r)
         / factorial(u)
         / factorial(i - 2 * r - 2 * u))
    )

    return arrA
    
def nuclear_integral(l1, m1, n1, l2, m2, n2, Z, alpha1, alpha2, N1, N2):
    """
    Compute the nuclear attraction integral between two Gaussian Type Orbitals (GTOs).

    Parameters
    ----------
    l1, m1, n1 : int
        GTO numbers for the angular momentum of the first GTO in the x, y, and z directions, respectively.
    l2, m2, n2 : int
        GTO numbers for the angular momentum of the second GTO in the x, y, and z directions, respectively.
    Z : float
        Nuclear charge of the atom or molecule.
    alpha1 : float
        Exponent of the first Gaussian Type Orbital.
    alpha2 : float
        Exponent of the second Gaussian Type Orbital.
    N1 : float
        Normalization constant of the first Gaussian Type Orbital.
    N2 : float
        Normalization constant of the second Gaussian Type Orbital.

    Returns
    -------
    float
        The value of the nuclear attraction integral.
    """
    # Combined Gaussian exponent
    gamma = alpha1 + alpha2

    # Compute the A arrays for x, y, and z directions
    Ax = A_array(l1, l2, gamma)
    Ay = A_array(m1, m2, gamma)
    Az = A_array(n1, n2, gamma)

    # Create index arrays for Ax, Ay, and Az
    i, j, k = np.array(
        np.meshgrid(
            np.arange(l1 + l2 + 1),
            np.arange(m1 + m2 + 1),
            np.arange(n1 + n2 + 1),
            indexing='ij'
        )
    ).reshape(3, -1)

    # Compute the Boys function terms
    boys = np.sum(1 / (2 * (i + j + k) + 1) * Ax[i] * Ay[j] * Az[k])

    # Compute the nuclear attraction integral
    return -Z * N1 * N2 * np.pi / gamma * 2 * boys