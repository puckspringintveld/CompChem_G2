import numpy as np
from pylebedev import PyLebedev

def Gauss_Lebedev_Chebychev(Radialpoints, Lebedevorder):
    rm = 0.35  # Scaling factor for radial coordinates

    # Build the Gauss-Chebyshev radial grid
    z = np.arange(1, Radialpoints + 1)  # Indices for radial grid points
    x = np.cos(np.pi / (Radialpoints + 1) * z)  # Cosine mapping for Chebyshev points
    r = rm * (1 + x) / (1 - x)  # Radial transformation to stretch grid points
    wr = (np.pi / (Radialpoints + 1) * np.sin(np.pi / (Radialpoints + 1) * z)**2
            * 2.0 * rm / (np.sqrt(1 - x**2) * (1 - x)**2))  # Radial weights

    # Get Lebedev quadrature points and weights
    leblib = PyLebedev()  # Initialize the Lebedev quadrature library
    p, wl = leblib.get_points_and_weights(Lebedevorder)  # Obtain angular points and weights

    # Construct the full 3D grid by combining radial and angular grids
    gridpts = np.outer(r, p).reshape((-1, 3))  # Combine radial and angular grid points

    # Separate x, y, z coordinates from the grid
    x = gridpts[:, 0]
    y = gridpts[:, 1]
    z = gridpts[:, 2]

    gridw = np.outer(wr * r**2, wl).flatten()  # Combine radial and angular weights

    return x, y, z, gridw

def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x : numpy.ndarray or float
        Cartesian x-coordinate(s).
    y : numpy.ndarray or float
        Cartesian y-coordinate(s).
    z : numpy.ndarray or float
        Cartesian z-coordinate(s).

    Returns
    -------
    r : numpy.ndarray or float
        Radial distance from the origin.
    theta : numpy.ndarray or float
        Polar angle (in radians), measured from the positive z-axis.
    phi : numpy.ndarray or float
        Azimuthal angle (in radians), measured counterclockwise from the positive x-axis.
    """
    # Compute the radial distance
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Avoid division by zero for points at the origin
    r[r == 0] = 1e-10
    
    # Compute the polar angle (theta)
    theta = np.arccos(z / r)
    
    # Compute the azimuthal angle (phi)
    phi = np.arctan2(y, x)
    
    return r, theta, phi