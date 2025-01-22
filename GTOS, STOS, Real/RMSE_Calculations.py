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

Radialpoints, Lebedevorder = 128, 59

x, y, z, gridw = Gauss_Lebedev_Chebychev(Radialpoints, Lebedevorder)

# Prevent division by zero in the GTO computation
epsilon = 1e-10
x = np.where(x == 0, epsilon, x)
y = np.where(y == 0, epsilon, y)
z = np.where(z == 0, epsilon, z)

R, Theta, Phi = cartesian_to_spherical(x, y, z)

def main():
    STO_RMSE = pd.read_excel('STO_RMSE.xlsx').to_numpy() 
    STO_Data = pd.read_excel('coefficient_matrix_STO.xlsx').to_numpy()  

    GTO_RMSE = pd.read_excel('GTO_RMSE.xlsx').to_numpy() 
    GTO_Data = pd.read_excel('coefficient_matrix_GTO.xlsx').to_numpy()  

    for i in range(30):
        print(f"RMSE calcualtion for GTO and STO orbitals {i + 1} out of 30")

        Psi_STO = generate_orbitals_for_STO(6, STO_Data[0,:], STO_Data[2:,i])
        Psi_GTO = generate_orbitals_for_GTO(8, GTO_Data[0,:], GTO_Data[2:,i])

        Psi_Real_STO = wavefunction(int(STO_RMSE[0,i + 1]), int(STO_RMSE[1,i + 1]), int(STO_RMSE[2,i + 1]), R, Theta, Phi)
        Psi_Real_GTO = wavefunction(int(GTO_RMSE[0,i + 1]), int(GTO_RMSE[1,i + 1]), int(GTO_RMSE[2,i + 1]), R, Theta, Phi)

        STO_RMSE[3,i + 1] = np.sqrt(np.sum((Psi_STO**2 - Psi_Real_STO**2)**2)/len(R))
        GTO_RMSE[3,i + 1] = np.sqrt(np.sum((Psi_GTO**2 - Psi_Real_GTO**2)**2)/len(R))

        print(f"\t RMSE STO: {STO_RMSE[3,i + 1]}, RMSE GTO: {GTO_RMSE[3,i + 1]}")

        STO_RMSE[4,i + 1] = 4 * np.pi * np.sum(Psi_STO**2 * gridw)
        GTO_RMSE[4,i + 1] = 4 * np.pi * np.sum(Psi_GTO**2 * gridw)

        STO_RMSE[5,i + 1] = 100 * (STO_Data[1,i] - (-1/STO_RMSE[0,i + 1]**2/2))/(-1/STO_RMSE[0,i + 1]**2/2)
        GTO_RMSE[5,i + 1] = 100 * (GTO_Data[1,i] - (-1/GTO_RMSE[0,i + 1]**2/2))/(-1/GTO_RMSE[0,i + 1]**2/2)
    
    STO_RMSE = pd.DataFrame(STO_RMSE)
    STO_RMSE.to_excel("STO_RMSE.xlsx", index=False)

    GTO_RMSE = pd.DataFrame(GTO_RMSE)
    GTO_RMSE.to_excel("GTO_RMSE.xlsx", index=False)


def generate_orbitals_for_STO(Highest_Principle_Quantum_Number, Zetas, Coeffients):

    Psi = np.zeros(len(R))

    n, l, m = generate_quantum_numbers(Highest_Principle_Quantum_Number)

    for i in range(len(Coeffients)):
        Psi += Coeffients[i] * STO(n[i], l[i], m[i], Zetas[i], R, Theta, Phi)
    
    return Psi
import numpy as np

def generate_orbitals_for_GTO(Highest_Order, Alphas, coefficients):
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