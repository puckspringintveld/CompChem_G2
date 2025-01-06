import numpy as np
from scipy.special import factorial

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