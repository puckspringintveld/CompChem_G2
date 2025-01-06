# importing general python modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pytessel import PyTessel
import os 

# importing function from self created files
from gaussian_type_orbitals import normalization_constant

def plot(Alphas, l, m, n, coefficients, energy, filename):
    grid = auto_adjust_grid(Alphas, l, m, n, coefficients, 1e-8)
    
    psi, _, _, z =psi_plot(Alphas, l, m, n, coefficients, grid)

    # Compute z indices for slices
    z_indices = np.linspace(0, len(z) - 1, 9).astype(int)
    
    # Set up the figure
    fig = plt.figure(figsize=(13, 12))
    gs = gridspec.GridSpec(3, 3, wspace=0.3, hspace=0.3, right=0.85)
    extent = [-grid, grid, -grid, grid]
    limit = np.max(np.abs(psi))

    # Create subplots
    for idx, z_index in enumerate(z_indices):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(
            psi[:, :, z_index],
            origin='lower',
            extent=extent,
            cmap='PiYG',
            vmin=-limit,
            vmax=limit
        )
        ax.set_title(f"XY Plane at Z = {z[z_index]:.2f} (a.u.)")
        ax.set_xlabel("x (a.u.)")
        ax.set_ylabel("y (a.u.)")
    
    # Add a single colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Wavefunction Value")
    fig.suptitle(f"Energy Found: {energy:.8f} Ht", fontsize=16, y=0.92)
    plt.show()
    
    isovalue = 0.05 * np.max(psi**2)
    unitcell = np.diag(np.ones(3) * grid * 2)
    
    os.makedirs("Iso_Surfaces", exist_ok=True)  # Create folder if it doesn't exist

    pytessel = PyTessel()
    vertices, normals, indices = pytessel.marching_cubes(psi.flatten(), psi.shape, unitcell.flatten(), isovalue)
    fname = os.path.join("Iso_Surfaces", filename + '_pos.ply')  # Save in the folder
    pytessel.write_ply(fname, vertices, normals, indices)

    vertices, normals, indices = pytessel.marching_cubes(psi.flatten(), psi.shape, unitcell.flatten(), -isovalue)
    fname = os.path.join("Iso_Surfaces", filename + '_neg.ply')  # Save in the folder
    pytessel.write_ply(fname, vertices, normals, indices)

def psi_plot(Alphas, l, m, n, coefficients, grid):
    x = np.linspace(-grid, grid, 101)
    y = np.linspace(-grid, grid, 101)
    z = np.linspace(-grid, grid, 101)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')  # Proper axis alignment

    # Ensure proper reshaping for broadcasting
    Alphas = np.array(Alphas)[:, None, None, None]  # Shape (N, 1, 1, 1)
    l = np.array(l)[:, None, None, None]            # Shape (N, 1, 1, 1)
    m = np.array(m)[:, None, None, None]            # Shape (N, 1, 1, 1)
    n = np.array(n)[:, None, None, None]            # Shape (N, 1, 1, 1)
    coefficients = np.array(coefficients)[:, None, None, None]  # Shape (N, 1, 1, 1)

    # Add an axis for grid arrays to match the leading dimension (N)
    xx = xx[None, :, :, :]  # Shape (1, 101, 101, 101)
    yy = yy[None, :, :, :]  # Shape (1, 101, 101, 101)
    zz = zz[None, :, :, :]  # Shape (1, 101, 101, 101)

    # Numerically calculate psi
    psi_plot = np.sum(
        coefficients
        * normalization_constant(Alphas, l, m, n)  # Shape (N, 1, 1, 1)
        * (xx ** l) * (yy ** m) * (zz ** n)        # Shape (N, 101, 101, 101)
        * np.exp(-Alphas * (xx**2 + yy**2 + zz**2)),  # Shape (N, 101, 101, 101)
        axis=0  # Sum over Gaussian components
    )
    
    return psi_plot, x, y, z

def auto_adjust_grid(Alphas, l, m, n, coefficients, threshold_factor):
    """
    Automatically adjusts the grid size based on the wavefunction values.
    
    Parameters:
        psi_plot (ndarray): The computed wavefunction on the grid.
        x, y, z (ndarray): The grid points in each dimension.
        threshold_factor (float): Factor for determining significant values (default: 1e-3).
        padding (float): Padding around the bounding box (default: 2 units).
    
    Returns:
        Tuple: New ranges for x, y, z.
    """
    # Define the spatial grid
    psi, x, y, z = psi_plot(Alphas, l, m, n, coefficients, 100)
    
    threshold = threshold_factor * np.max(psi**2)
    significant_mask = np.abs(psi**2) > threshold

    # Find the bounds of significant values
    x_indices, y_indices, z_indices = np.where(significant_mask)
    x_min, x_max = x[x_indices.min()], x[x_indices.max()]
    y_min, y_max = y[y_indices.min()], y[y_indices.max()]
    z_min, z_max = z[z_indices.min()], z[z_indices.max()]

    return max(abs(np.array([x_min, x_max, y_min, y_max, z_min, z_max])))

