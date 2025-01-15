# importing general python modules
import pandas as pd
import numpy as np

# importing self created functions
from gaussian_type_orbitals import generate_lmn, normalization_constant
from slater_type_orbitals import STO, generate_quantum_numbers
from coordinates import cartesian_to_spherical, Gauss_Lebedev_Chebychev
from analytical import wavefunction
import matplotlib.pyplot as plt
from pylebedev import PyLebedev

def main():
    Radialpoints, Lebedevorder = 2048, 59 
    STO_RMSE = pd.read_excel('STO_RMSE.xlsx').to_numpy() 
    STO_Data = pd.read_excel('coefficient_matrix_STO.xlsx').to_numpy()  

    GTO_RMSE = pd.read_excel('GTO_RMSE.xlsx').to_numpy() 
    GTO_Data = pd.read_excel('coefficient_matrix_GTO.xlsx').to_numpy()  

    # Get Lebedev quadrature points and weights
    leblib = PyLebedev()  # Initialize the Lebedev quadrature library
    p, _ = leblib.get_points_and_weights(Lebedevorder)  # Obtain angular points and weights

    rn = 0.35
    for i in range(30):
        print(f"RMSE calcualtion for GTO and STO orbitals {i + 1} out of 30")

        r = np.linspace(0, 100, Radialpoints)
        # Construct the full 3D grid by combining radial and angular grids
        gridpts = np.outer(r, p).reshape((-1, 3))  # Combine radial and angular grid points

        # Separate x, y, z coordinates from the grid
        x = gridpts[:, 0]
        y = gridpts[:, 1]
        z = gridpts[:, 2]

        # Prevent division by zero in the GTO computation
        epsilon = 1e-10
        x = np.where(x == 0, epsilon, x)
        y = np.where(y == 0, epsilon, y)
        z = np.where(z == 0, epsilon, z)

        R, Theta, Phi = cartesian_to_spherical(x, y, z)

        Psi_STO = generate_orbitals_for_STO(6, STO_Data[0,:], STO_Data[2:,i], R, Theta, Phi)
        Psi_GTO = generate_orbitals_for_GTO(8, GTO_Data[0,:], GTO_Data[2:,i], x, y, z)

        Psi_Real_STO = wavefunction(int(STO_RMSE[0,i + 1]), int(STO_RMSE[1,i + 1]), int(STO_RMSE[2,i + 1]), R, Theta, Phi)
        Psi_Real_GTO = wavefunction(int(GTO_RMSE[0,i + 1]), int(GTO_RMSE[1,i + 1]), int(GTO_RMSE[2,i + 1]), R, Theta, Phi)

        STO_RMSE[3,i + 1] = np.sqrt(np.sum((Psi_Real_STO**2 - Psi_STO**2)**2) / len(Psi_Real_STO))
        GTO_RMSE[3,i + 1] = np.sqrt(np.sum((Psi_Real_GTO**2 - Psi_GTO**2)**2) / len(Psi_Real_GTO))

        rm = 0.35  # Scaling factor for radial coordinates

        # Build the Gauss-Chebyshev radial grid
        z = np.arange(1, Radialpoints + 1)  # Indices for radial grid points
        x = np.cos(np.pi / (Radialpoints + 1) * z)  # Cosine mapping for Chebyshev points
        r = rm * (1 + x) / (1 - x)  # Radial transformation to stretch grid points

        gridpts = np.outer(r, p).reshape((-1, 3))  # Combine radial and angular grid points

        # Separate x, y, z coordinates from the grid
        x = gridpts[:, 0]
        y = gridpts[:, 1]
        z = gridpts[:, 2]

        # Prevent division by zero in the GTO computation
        epsilon = 1e-10
        x = np.where(x == 0, epsilon, x)
        y = np.where(y == 0, epsilon, y)
        z = np.where(z == 0, epsilon, z)

        R, Theta, Phi = cartesian_to_spherical(x, y, z)

        Psi_STO = generate_orbitals_for_STO(6, STO_Data[0,:], STO_Data[2:,i], R, Theta, Phi)
        Psi_GTO = generate_orbitals_for_GTO(8, GTO_Data[0,:], GTO_Data[2:,i], x, y, z)

        Psi_Real_STO = wavefunction(int(STO_RMSE[0,i + 1]), int(STO_RMSE[1,i + 1]), int(STO_RMSE[2,i + 1]), R, Theta, Phi)
        Psi_Real_GTO = wavefunction(int(GTO_RMSE[0,i + 1]), int(GTO_RMSE[1,i + 1]), int(GTO_RMSE[2,i + 1]), R, Theta, Phi)

        STO_RMSE[4,i + 1] = np.sqrt(np.sum((Psi_Real_STO**2 - Psi_STO**2)**2) / len(Psi_Real_STO))
        GTO_RMSE[4,i + 1] = np.sqrt(np.sum((Psi_Real_GTO**2 - Psi_GTO**2)**2) / len(Psi_Real_GTO))

    
    STO_RMSE = pd.DataFrame(STO_RMSE)
    STO_RMSE.to_excel("STO_RMSE.xlsx", index=False)

    GTO_RMSE = pd.DataFrame(GTO_RMSE)
    GTO_RMSE.to_excel("GTO_RMSE.xlsx", index=False)


def generate_orbitals_for_STO(Highest_Principle_Quantum_Number, Zetas, Coeffients, R, Theta, Phi):

    Psi = np.zeros(len(R))

    n, l, m = generate_quantum_numbers(Highest_Principle_Quantum_Number)

    for i in range(len(Coeffients)):
        Psi += Coeffients[i] * STO(n[i], l[i], m[i], Zetas[i], R, Theta, Phi)
    
    return Psi
import numpy as np

def generate_orbitals_for_GTO(Highest_Order, Alphas, coefficients, x, y, z):
    # Generate l, m, n (ensure generate_lmn function is defined)
    l, m, n = generate_lmn(Highest_Order)

    # Convert inputs to NumPy arrays
    Alphas = np.array(Alphas)[:, None]  # Shape (N, 1)
    coefficients = np.array(coefficients)[:, None]  # Shape (N, 1)
    l = np.array(l)[:, None]  # Shape (N, 1)
    m = np.array(m)[:, None]  # Shape (N, 1)
    n = np.array(n)[:, None]  # Shape (N, 1)

    # Reshape x, y, z for broadcasting
    xx = x[None, :]  
    yy = y[None, :]  
    zz = z[None, :] 

    # Compute Psi using np.sum
    Psi = np.sum(
        coefficients
        * normalization_constant(Alphas, l, m, n)  # Normalization constant
        * (xx ** l) * (yy ** m) * (zz ** n)           # Angular momentum terms
        * np.exp(-Alphas * (xx**2 + yy**2 + zz**2)),  # Gaussian exponentials
        axis=0  # Sum over all Gaussian terms
    )

    return Psi

if __name__ == "__main__":
    main()