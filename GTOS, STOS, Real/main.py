# importing general python modules
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import time 

# setting the current directory to where the file is located
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# importing functions from self created files
from slater_type_orbitals import generate_quantum_numbers
from gaussian_type_orbitals import generate_lmn
from linear_variation import Linear_Variation, energy_function, Matrixes_Analytically_Pyqint, Matrixes_Numerically_STO, Matrixes_Analytically, Matrixes_Numerically
from plotting import plot

def main():
    # defining constants
    Radialpoints = 0 # number of radial sampeling points
    Lebedevorder = 0 # Levedev-Chebychev order used
    Z = 1 # nuclear charge of hydrogen

    print("")
    print("Choosing Method \n \t 0: GTO Numerical Integration Via Gauss-Chebyshev-Lebedev \n \t 1: GTO Analytical Solutions in Python \n \t 2: GTO Analytical Solutions via Pyqint \n \t 3: STO Numerical Integration Via Gauss-Chebyshev-Lebedev \n")
    
    method = int(input("Enter your method of choice (default: 2): ") or 2)
    
    print(f"\t You chose method {method}")
    if method == 0 or method == 3:
        print("\t The numerical integration method requires a choice of sampling points")
        Radialpoints = int(input("\t Enter your choice of radial sampling points (default: 32): ") or 32)
        Lebedevorder = int(input("\t Enter your choice of Lebedev order (angular sampling points) (default: 17): ") or 17)
    
    print("")
    
    if method == 3:
        order_slater_orbitals = int(input("Enter the desired higest principle quantum number of the Slater Type Orbitals to use as a Basis Set (default: 4): ") or 4)
        n, l, m = generate_quantum_numbers(order_slater_orbitals)
        print(f"\t You chose the highest principle quantum number of the STO's to be: {order_slater_orbitals}")    
        print(f"\t A max principle quantum number of {order_slater_orbitals} corresponds to a Basis Set size of: {len(l)} \n")

        N = len(n)
        X0 = 1/n #good initial guess for alpha
    else:
        order_gaussian_orbitals = int(input("Enter the desired higest order of Gaussian Type Orbitals to use as a Basis Set (default: 4): ") or 4)
        l, m, n = generate_lmn(order_gaussian_orbitals)
        print(f"\t You chose the order of GTO's to be: {order_gaussian_orbitals}")    
        print(f"\t A max order of {order_gaussian_orbitals} GTO's corresponds to a Basis Set size of: {len(l)} \n")
    
        # generating a "good" initial guesses for the alphas for optimizing
        N = len(l)
        Beta = 1.2
        Alpha0 = 0.2
        X0 = np.array([Alpha0 * Beta ** -(i - 1)/i for i in range(1, N + 1)])    

    start_time = time.time()  # Record the start time
    
    # in principle a minimization is not needed but it does improve the results
    result = minimize(energy_function, x0=X0, bounds=[(1e-8, 1e8)], args=(Lebedevorder, Radialpoints, l, m, n, Z, method))
    optimized_alphas = result.x # extracting the optimized alphas
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time # Record the elapsed time

    print(f"Elapsed time for minimization of the energies to find the best alphas: {elapsed_time:.6f} seconds \n")
    
    print("Results")

    print(f"Optimized Alphas: {optimized_alphas}")

    # Compute the final energy using the optimized alpha using fastest method
    if method == 0:
        H, S =  Matrixes_Numerically(optimized_alphas, Lebedevorder, Radialpoints, l, m, n, Z)
    elif method == 1:
        H, S = Matrixes_Analytically(optimized_alphas, l, m, n, Z)
    elif method == 2:
        H, S = Matrixes_Analytically_Pyqint(optimized_alphas, l, m, n, Z)
    elif method == 3:
        H, S = Matrixes_Numerically_STO(n, l, m, optimized_alphas, Radialpoints, Lebedevorder)

    energy, coefficients = Linear_Variation(H, S)
    print(np.shape(energy))

    # saving coefficient matrix
    coefficients_df = pd.DataFrame(coefficients)
    if method == 3:
        coefficients_df.to_excel("coefficient_matrix_STO.xlsx", index=False)
    else:
        coefficients_df.to_excel("coefficient_matrix_GTO.xlsx", index=False)

    print(f"Final Energies: {energy} \n")
    
    print("plotting")
    print("\t The contour plots of the optimized hydrogen-like orbitals will be plotted and saved in a .png format")
    print("\t The isosurface files of the hydrogen-like orbitals are also saved in a .ply format for an isovalue corresponding to 95% electron density")

    for i in range(N):
        print(f"\t Plotting orbital {i + 1} out of {N}")
        plot(optimized_alphas, l, m, n, coefficients[:, i], energy[i], f"orbital_{i}", method)
    
if __name__ == "__main__":
    main()