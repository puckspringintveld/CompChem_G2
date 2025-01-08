# importing general python modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pytessel import PyTessel
import os 

# importing function from self created files
from gaussian_type_orbitals import normalization_constant

def plot(Alphas, l, m, n, coefficients, energy, filename):
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
    grid = auto_adjust_grid(Alphas, l, m, n, coefficients, 1e-4)
    
    # Generate wavefunction (psi) and additional outputs based on the grid
    psi, _, _, z = psi_plot(Alphas, l, m, n, coefficients, grid)
    
    # Compute indices for evenly spaced slices along the z-axis
    z_indices = np.linspace(0, len(z) - 1, 9).astype(int)
    
    # Ensure the folder for contour plots exists
    os.makedirs("Contour_Plots", exist_ok=True)
    fname = os.path.join("Contour_Plots", filename + ".png")  # contour plot filename

    # Create a new figure for the plot
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
        # Display the XY plane slice of the wavefunction at the given z-index
        im = ax.imshow(
            psi[:, :, z_index],
            origin='lower',
            extent=extent,
            cmap='PiYG',  # Use PiYG colormap
            vmin=-limit,  # Set color scale minimum
            vmax=limit    # Set color scale maximum
        )
        # Add a title and axis labels to the subplot
        ax.set_title(f"XY Plane at Z = {z[z_index]:.2f} (a.u.)")
        ax.set_xlabel("x (a.u.)")
        ax.set_ylabel("y (a.u.)")
    
    # Add a single colorbar to the figure
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # Define colorbar position
    fig.colorbar(im, cax=cbar_ax, label="Wavefunction Value")
    # Add a figure-wide title showing the computed energy
    fig.suptitle(f"Energy Found: {energy:.8f} Ht", fontsize=16, y=0.92)
    
    fig.savefig(fname, dpi=300, bbox_inches="tight")  # Ensure high-quality saving
    # Display the plots
    plt.show()
    
    # Adjust the grid based on the provided parameters and a tolerance of 1e-4
    grid = auto_adjust_grid(Alphas, l, m, n, coefficients, 1e-10)
    
    # Generate wavefunction (psi) and additional outputs based on the grid
    psi, _, _, z = psi_plot(Alphas, l, m, n, coefficients, grid)
    # numerical error causes some 1e-16 imaginary part to remain even after the tesseral tranformation
    psi = np.real(psi)
    
    # Compute the electron density (normalized wavefunction squared)
    density = np.real(psi)**2
 
    # Flatten the density array for processing
    flat_density = density.flatten()
 
    # Sort density values in descending order
    sorted_density = np.sort(flat_density)[::-1]
 
    # Compute the cumulative sum
    cumulative_density = np.cumsum(sorted_density)
    cumulative_density/= cumulative_density[-1]
 
    # Find the isovalue corresponding to 95% electron density
    isovalue_index = np.searchsorted(cumulative_density, 0.95)
    isovalue = sorted_density[isovalue_index]
    # Create a unit cell based on the grid size (arbitrarily)
    unitcell = np.diag(np.ones(3) * 1)
    
    # Ensure the folder for isosurface files exists
    os.makedirs("Iso_Surfaces", exist_ok=True)

    # Initialize PyTessel for isosurface generation
    pytessel = PyTessel()
    # Generate and save the positive isosurface
    vertices, normals, indices = pytessel.marching_cubes(psi.flatten(), psi.shape, unitcell.flatten(), isovalue)
    fname = os.path.join("Iso_Surfaces", filename + '_pos.ply')  # Positive isosurface filename
    pytessel.write_ply(fname, vertices, normals, indices)

    # Generate and save the negative isosurface
    vertices, normals, indices = pytessel.marching_cubes(psi.flatten(), psi.shape, unitcell.flatten(), -isovalue)
    fname = os.path.join("Iso_Surfaces", filename + '_neg.ply')  # Negative isosurface filename
    pytessel.write_ply(fname, vertices, normals, indices)

def psi_plot(Alphas, l, m, n, coefficients, grid):
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
    x = np.linspace(-grid, grid, 101)
    y = np.linspace(-grid, grid, 101)
    z = np.linspace(-grid, grid, 101)

    # Create 3D mesh grids for the x, y, and z coordinates
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')  # Proper axis alignment

    # Ensure proper reshaping for broadcasting Gaussian parameters
    Alphas = np.array(Alphas)[:, None, None, None]  # Shape (N, 1, 1, 1)
    l = np.array(l)[:, None, None, None]            # Shape (N, 1, 1, 1)
    m = np.array(m)[:, None, None, None]            # Shape (N, 1, 1, 1)
    n = np.array(n)[:, None, None, None]            # Shape (N, 1, 1, 1)
    coefficients = np.array(coefficients)[:, None, None, None]  # Shape (N, 1, 1, 1)

    # Add an extra dimension for grid arrays to match Gaussian parameters
    xx = xx[None, :, :, :]  # Shape (1, 101, 101, 101)
    yy = yy[None, :, :, :]  # Shape (1, 101, 101, 101)
    zz = zz[None, :, :, :]  # Shape (1, 101, 101, 101)

    # Compute the wavefunction values on the 3D grid
    psi_plot = np.sum(
        coefficients
        * normalization_constant(Alphas, l, m, n)  # Compute normalization constant
        * (xx ** l) * (yy ** m) * (zz ** n)        # Apply angular momentum terms
        * np.exp(-Alphas * (xx**2 + yy**2 + zz**2)),  # Apply Gaussian exponentials
        axis=0  # Sum contributions from all basis functions
    )
    
    # Return the computed wavefunction and the grid coordinates
    return psi_plot, x, y, z


def auto_adjust_grid(Alphas, l, m, n, coefficients, threshold_factor):
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
    psi, x, y, z = psi_plot(Alphas, l, m, n, coefficients, 100)
    
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


