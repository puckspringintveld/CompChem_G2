import numpy as np

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