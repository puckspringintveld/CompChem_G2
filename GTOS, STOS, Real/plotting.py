# importing general python modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pytessel import PyTessel
import os 

# importing function from self created files
from gaussian_type_orbitals import GTO
from slater_type_orbitals import STO
from coordinates import cartesian_to_spherical

def plot(Alphas, l, m, n, coefficients, energy, filename, method, gridsize):
    """
    Visualize wavefunction slices in the XY plane and generate isosurface files.

    Parameters
    ----------
    Alphas : numpy.ndarray
        Array of exponents used in the wavefunction computation.
    l : int
        Angular momentum gto number in the x-direction.
    m : int
        Angular momentum gto number in the y-direction.
    n : int
        Angular momentum gto number in the z-direction.
    coefficients : numpy.ndarray
        Coefficients for the basis functions in the wavefunction.
    energy : float
        Computed energy value of the wavefunction (in Hartree units).
    filename : str
        Base name for saving isosurface files.

    Returns
    -------
    None
        This function generates plots and saves isosurface files, but does not return any values.
    """
    # Adjust the grid based on the provided parameters and a tolerance of 1e-4
    grid = auto_adjust_grid(Alphas, l, m, n, coefficients, 1e-4, method)
    
    # Generate wavefunction (psi) and additional outputs based on the grid
    psi, x, y, z = psi_plot(Alphas, l, m, n, coefficients, grid, method, 101)
    
    # Compute indices for evenly spaced slices along the z-axis
    z_indices = np.linspace(0, len(z) - 1, 9).astype(int)
    
    # Ensure the folder for contour plots exists
    if method == 3:
        os.makedirs("Contour_Plots_STO", exist_ok=True)
        fname = os.path.join("Contour_Plots_STO", filename + ".png")  # contour plot filename
        os.makedirs("Iso_Surfaces_STO", exist_ok=True)
        fnamepos = os.path.join("Iso_Surfaces_STO", filename + '_pos.ply')  # Positive isosurface filename
        fnameneg = os.path.join("Iso_Surfaces_STO", filename + '_neg.ply')  # Negative isosurface filename
    else:
        os.makedirs("Contour_Plots_GTO", exist_ok=True)
        fname = os.path.join("Contour_Plots_GTO", filename + ".png")  # contour plot filename
        os.makedirs("Iso_Surfaces_GTO", exist_ok=True)
        fnamepos = os.path.join("Iso_Surfaces_GTO", filename + '_pos.ply')  # Positive isosurface filename
        fnameneg = os.path.join("Iso_Surfaces_GTO", filename + '_neg.ply')  # Negative isosurface filename

    # Create a new figure for the contour plots
    fig = plt.figure(figsize=(13, 12))
    # Define a 3x3 grid layout for subplots with specified spacing
    gs = gridspec.GridSpec(3, 3, wspace=0.3, hspace=0.3, right=0.85)
    # Set plot extent and limits based on the grid
    extent = [-grid, grid, -grid, grid]
    limit = np.max(np.abs(psi))

    # Loop over the z indices to create individual subplots
    for idx, z_index in enumerate(z_indices):
        # Calculate row and column for the subplot
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])  # Add a subplot

        # Create a contour plot of the wavefunction at the given z-index
        levels = np.linspace(-limit, limit, 51, endpoint=True)
        im = ax.contourf(
            x, y, psi[:, :, z_index],
            levels=levels,
            cmap='seismic',  # Use PiYG colormap
            extent=extent
        )
        
        # Add a title and axis labels to the subplot
        ax.set_title(f"XY Plane at Z = {z[z_index]:.2f} (a.u.)")
        ax.set_xlabel("x (a.u.)")
        ax.set_ylabel("y (a.u.)")
        ax.set_aspect('equal', 'box')

    # Add a single colorbar to the figure
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # Define colorbar position
    fig.colorbar(im, cax=cbar_ax, label="Wavefunction Value")
    # Add a figure-wide title showing the computed energy
    fig.suptitle(f"Energy Found: {energy:.8f} Ht", fontsize=16, y=0.92)
    fig.savefig(fname, dpi=300, bbox_inches="tight")  # Ensure high-quality saving
    plt.close()

    # Adjust the grid based on the provided parameters and a tolerance of 1e-4
    grid = auto_adjust_grid(Alphas, l, m, n, coefficients, 1e-8, method)

    # Generate wavefunction (psi) and additional outputs based on the grid
    psi, _, _, z = psi_plot(Alphas, l, m, n, coefficients, grid, method, gridsize)
    dz = (z[1] - z[0])
    
    # Compute the electron density (normalized wavefunction squared)
    density = psi**2 * dz**3

    # Flatten the density array for processing
    flat_density = density.flatten()

    # Sort density values in descending order
    sorted_density = np.sort(flat_density)[::-1]

    # Compute the cumulative sum
    cumulative_density = np.cumsum(sorted_density)
    print(f"\t Density: {cumulative_density[-1]}")

    # Find the isovalue corresponding to 90% electron density
    isovalue_index = np.searchsorted(cumulative_density, 0.90)
    isovalue = sorted_density[isovalue_index]
    
    # Create a unit cell based on the grid size
    unitcell = np.diag(np.ones(3) * grid/10)
    
    # Initialize PyTessel for isosurface generation
    pytessel = PyTessel()
    # Generate and save the positive isosurface
    vertices, normals, indices = pytessel.marching_cubes(psi.flatten(), psi.shape, unitcell.flatten(), isovalue)
    pytessel.write_ply(fnamepos, vertices, normals, indices)

    # Generate and save the negative isosurface
    vertices, normals, indices = pytessel.marching_cubes(psi.flatten(), psi.shape, unitcell.flatten(), -isovalue)
    pytessel.write_ply(fnameneg, vertices, normals, indices)

def psi_plot(Alphas, l, m, n, coefficients, grid, method, gridsize):
    """
    Calculate the wavefunction values on a 3D grid for visualization.

    Parameters
    ----------
    Alphas : numpy.ndarray
        Array of exponents used in the wavefunction computation.
    l : int
        Angular momentum gto number in the x-direction.
    m : int
        Angular momentum gto number in the y-direction.
    n : int
        Angular momentum gto number in the z-direction.
    coefficients : numpy.ndarray
        Coefficients for the basis functions in the wavefunction.
    grid : float
        The extent of the grid in atomic units (a.u.).

    Returns
    -------
    psi_plot : numpy.ndarray
        3D array containing the computed wavefunction values on the grid.
    x : numpy.ndarray
        1D array of x-coordinates for the grid.
    y : numpy.ndarray
        1D array of y-coordinates for the grid.
    z : numpy.ndarray
        1D array of z-coordinates for the grid.
    """
    # Define a linear grid of points in the x, y, and z directions
    x = np.linspace(-grid, grid, gridsize)
    y = np.linspace(-grid, grid, gridsize)
    z = np.linspace(-grid, grid, gridsize)

    # Create 3D mesh grids for the x, y, and z coordinates
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')  # Proper axis alignment

    psi_plot = np.zeros((gridsize, gridsize, gridsize))

    # Compute the wavefunction values on the 3D grid
    if method == 3:
        R, Theta, Phi = cartesian_to_spherical(xx, yy, zz)
        for i in range(0, len(coefficients)):
            psi_plot += coefficients[i] * STO(n[i], l[i], m[i], Alphas[i], R, Theta, Phi)
    else:
        for i in range(0, len(coefficients)):
            psi_plot += coefficients[i] * GTO(xx, yy, zz, Alphas[i], l[i], m[i], n[i])
    
    # Return the computed wavefunction and the grid coordinates
    return psi_plot, x, y, z


def auto_adjust_grid(Alphas, l, m, n, coefficients, threshold_factor, method):
    """
    Automatically adjust the spatial grid size based on the significant regions of the wavefunction.

    Parameters
    ----------
    Alphas : numpy.ndarray
        Array of exponents used in the wavefunction computation.
    l : int
        Angular momentum gto number in the x-direction.
    m : int
        Angular momentum gto number in the y-direction.
    n : int
        Angular momentum gto number in the z-direction.
    coefficients : numpy.ndarray
        Coefficients for the basis functions in the wavefunction.
    threshold_factor : float
        Factor to determine the significance threshold for the wavefunction values.

    Returns
    -------
    float
        The maximum absolute bound for the spatial grid based on significant wavefunction values.
    """
    # Generate an initial spatial grid with an arbitrary size of 100
    psi, x, y, z = psi_plot(Alphas, l, m, n, coefficients, 200, method, 100)
    
    # Compute the threshold for significant wavefunction values
    threshold = threshold_factor * np.max(psi**2)
    
    # Create a mask for regions where the wavefunction squared exceeds the threshold
    significant_mask = np.abs(psi**2) > threshold

    # Find the indices of significant regions along each axis
    x_indices, y_indices, z_indices = np.where(significant_mask)
    
    # Determine the minimum and maximum bounds for significant x values
    x_min, x_max = x[x_indices.min()], x[x_indices.max()]
    # Determine the minimum and maximum bounds for significant y values
    y_min, y_max = y[y_indices.min()], y[y_indices.max()]
    # Determine the minimum and maximum bounds for significant z values
    z_min, z_max = z[z_indices.min()], z[z_indices.max()]

    # Return the maximum absolute bound across all axes
    return max(abs(np.array([x_min, x_max, y_min, y_max, z_min, z_max])))