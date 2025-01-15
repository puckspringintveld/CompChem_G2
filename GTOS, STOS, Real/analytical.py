# importing general python modules
import numpy as np
from sympy.physics.hydrogen import Psi_nlm
from sympy import Symbol, lambdify
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pytessel import PyTessel

# importing self made functions
from coordinates import cartesian_to_spherical

# setting the current directory to where the file is located
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def main():
    print("")
    max_n = int(input("Enter up untill which principle quantum number to generate (default: 4): ") or 4)
    print("\t The contour plots of the optimized hydrogen-like orbitals will be plotted and saved in a .png format")
    print("\t The isosurface files of the hydrogen-like orbitals are also saved in a .ply format for an isovalue corresponding to 95% electron density")


    for n in range(1, max_n + 1):  # Principal quantum number (n >= 1)
        for l in range(0, n):  # Azimuthal quantum number (0 <= l < n)
            for m in range(-l, l + 1):  # Magnetic quantum number (-l <= m <= l)
                plot(n, l, m)
                print(f"\t plotting orbital n: {n}, l: {l}, m: {m}")

def plot(n, l, m):
    """
    Visualize the wavefunction for given principal quantum numbers and generate contour plots
    and isosurface files.

    Parameters
    ----------
    n : int
        Principal quantum number indicating the energy level of the wavefunction.
    l : int
        Angular momentum quantum number for the wavefunction.
    m : int
        Magnetic quantum number .

    Returns
    -------
    None
        Saves contour plots and isosurface files to the respective directories.
    """
    # Adjust grid size for visualization and compute the wavefunction
    grid = auto_adjust_grid(n, l, m, 1e-4)
    psi, x, y, z = psi_plot(n, l, m, grid)

    # Compute the energy of the wavefunction
    energy = -1 / n**2 / 2

    # Determine z-slice indices for visualization
    z_indices = np.linspace(0, len(z) - 1, 9).astype(int)

    # Ensure the folder for contour plots exists
    os.makedirs("Contour_Plots_Real", exist_ok=True)
    filename = f"{n}_{l}_{m}"
    fname = os.path.join("Contour_Plots_Real", filename + ".png")  # Contour plot filename

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
    fig.suptitle(f"Energy: {energy:.8f} Ht, n: {n}, l:{l}, m{m}", fontsize=16, y=0.92)
    fig.savefig(fname, dpi=300, bbox_inches="tight")  # Ensure high-quality saving
    plt.close()
    
    # Adjust grid size for visualization and compute the wavefunction
    grid = auto_adjust_grid(n, l, m, 1e-10)
    psi, _, _, z = psi_plot(n, l, m, grid)
    
    # Compute the electron density (normalized wavefunction squared)
    density = psi**2

    # Flatten the density array for processing
    flat_density = density.flatten()

    # Sort density values in descending order
    sorted_density = np.sort(flat_density)[::-1]

    # Compute the cumulative sum
    cumulative_density = np.cumsum(sorted_density)
    cumulative_density /= cumulative_density[-1]

    # Find the isovalue corresponding to 95% electron density
    isovalue_index = np.searchsorted(cumulative_density, 0.95)
    isovalue = sorted_density[isovalue_index]
    
    # Create a unit cell based on the grid size
    unitcell = np.diag(np.ones(3) * 1)

    # Ensure the folder for isosurface files exists
    os.makedirs("Iso_Surfaces_Real", exist_ok=True)

    # Initialize PyTessel for isosurface generation
    pytessel = PyTessel()

    # Generate and save the positive isosurface
    vertices, normals, indices = pytessel.marching_cubes(psi.flatten(), psi.shape, unitcell.flatten(), isovalue)
    fname = os.path.join("Iso_Surfaces_Real", filename + '_pos.ply')  # Positive isosurface filename
    pytessel.write_ply(fname, vertices, normals, indices)

    # Generate and save the negative isosurface
    vertices, normals, indices = pytessel.marching_cubes(psi.flatten(), psi.shape, unitcell.flatten(), -isovalue)
    fname = os.path.join("Iso_Surfaces_Real", filename + '_neg.ply')  # Negative isosurface filename
    pytessel.write_ply(fname, vertices, normals, indices)

def psi_plot(n, l, m, grid):
    """
    Compute and visualize the hydrogen-like wavefunction on a 3D Cartesian grid.

    Parameters
    ----------
    n : int
        Principal quantum number indicating the energy level.
    l : int
        Angular momentum quantum number.
    m : int
        Magnetic quantum number.
    grid : float
        Extent of the Cartesian grid in atomic units (a.u.).

    Returns
    -------
    numpy.ndarray
        Evaluated wavefunction values on the grid.
    numpy.ndarray
        1D array of x-coordinates for the grid.
    numpy.ndarray
        1D array of y-coordinates for the grid.
    numpy.ndarray
        1D array of z-coordinates for the grid.
    """
    # Define Cartesian grid
    x = np.linspace(-grid, grid, 101)
    y = np.linspace(-grid, grid, 101)
    z = np.linspace(-grid, grid, 101)
    xx, yy, zz = np.meshgrid(x, y, z)

    # Convert Cartesian coordinates to spherical coordinates
    R, Theta, Phi = cartesian_to_spherical(xx, yy, zz)

    return wavefunction(n, l, m, R, Theta, Phi), x, y, z

def wavefunction(n, l, m, R, Theta, Phi):
     # Define symbolic variables for the hydrogen-like wavefunction
    r = Symbol("r", positive=True)
    phi = Symbol("phi", real=True)
    theta = Symbol("theta", real=True)
    Z = Symbol("Z", positive=True, integer=True, nonzero=True)

    # Obtain the symbolic representation of the wavefunction
    psi_symbolic = Psi_nlm(n, l, m, r, phi, theta, Z)
    psi_symbolic_opposite_m = Psi_nlm(n, l, -m, r, phi, theta, Z)

    # Convert the symbolic wavefunction to a NumPy-compatible function
    psi_func = lambdify((r, theta, phi, Z), psi_symbolic, modules="numpy")
    psi_func_opposite_m = lambdify((r, theta, phi, Z), psi_symbolic_opposite_m, modules="numpy")

    # Apply tesseral harmonic transformation
    if m < 0:
        psi_m = psi_func(R, Theta, Phi, 1)
        psi_neg_m = psi_func_opposite_m(R, Theta, Phi, 1)
        psi = 1j * (psi_m - (-1)**m * psi_neg_m) / np.sqrt(2)
    elif m > 0:
        psi_m = psi_func(R, Theta, Phi, 1)
        psi_neg_m = psi_func_opposite_m(R, Theta, Phi, 1)
        psi = (psi_neg_m + (-1)**m * psi_m) / np.sqrt(2)
    else:
        psi = psi_func(R, Theta, Phi, 1)

    return np.real(psi)

def auto_adjust_grid(n, l, m, threshold_factor):
    """
    Automatically adjust the spatial grid size based on significant regions of the wavefunction.

    Parameters
    ----------
    n : int
        Principal quantum number indicating the energy level.
    l : int
        Angular momentum quantum number.
    m : int
        Magnetic quantum number.
    threshold_factor : float
        Factor to determine the significance threshold for the wavefunction values.

    Returns
    -------
    float
        The maximum absolute bound for the spatial grid, based on the significant wavefunction values.
    """
    # Generate an initial spatial grid with an arbitrary size of 100
    psi, x, y, z = psi_plot(n, l, m, 100)

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

if __name__ == "__main__":
    main()