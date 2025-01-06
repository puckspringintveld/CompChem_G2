# importing general python modules
import numpy as np
from scipy.optimize import minimize
import time 

# setting the current directory to where the file is located
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# importing functions from self created files
from gaussian_type_orbitals import generate_lmn
from linear_variation import Linear_Variation, energy_function, Matrixes_Analytically_Pyqint
from plotting import plot

def main():
    # defining constants
    Radialpoints = 0 # number of radial sampeling points
    Lebedevorder = 0 # Levedev-Chebychev order used
    Z = 1 # nuclear charge of hydrogen
    
    print("Choosing Method \n \t 0: Numerical Integration Via Gauss-Lebedev-Chebychev \n \t 1: Analytical Solutions in Python \n \t 2: Analytical Solutions via Pyqint \n")
    
    method = int(input("Enter your method of choice (default: 2): ") or 2)
    
    print(f"\t You chose method {method}")
    if method == 0:
        print("\t The numerical integration method requires a choice of sampling points")
        Radialpoints = int(input("\t Enter your choice of radial sampling points (default: 32): ") or 32)
        Lebedevorder = int(input("\t Enter your choice of Lebedev-Chebychev order (angular sampling points) (default: 17): ") or 17)
    
    print("")
        
    order_gaussian_orbitals = int(input("Enter the desired higest order of Gaussian Type Orbitals to use as a Basis Set (default: 4): ") or 4)

    print(f"\t You chose the order of GTO's to be: {order_gaussian_orbitals}")    
    
    l, m, n = generate_lmn(order_gaussian_orbitals)
    
    print(f"\t A max order of {order_gaussian_orbitals} GTO's corresponds to a Basis Set size of: {len(l)} \n")
    
    # generating a "good" initial guesses for the alphas for optimizing
    N = len(l)
    Beta = 1.2
    Alpha0 = 0.2
    X0 = np.array([Alpha0 * Beta ** -(i - 1)/i for i in range(1, N + 1)])    

    start_time = time.time()  # Record the start time
    
    result = minimize(energy_function, x0=X0, bounds=[(1e-8, 1e8)], args=(Lebedevorder, Radialpoints, l, m, n, Z, method))
    optimized_alphas = result.x # extracting the optimized alphas
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time # Record the elapsed time

    print(f"Elapsed of minimization of the energies to find the best alphas: {elapsed_time:.6f} seconds \n")
    
    print(f"Results")

    print(f"Optimized Alphas: {optimized_alphas}")

    # Compute the final energy using the optimized alpha using fastest method
    H, S = Matrixes_Analytically_Pyqint(optimized_alphas, l, m, n, Z)
    energy, coefficients = Linear_Variation(H, S)
    
    print(f"Final Energies: {energy} \n")
    
    print("plotting")
    print("\t The contour plots of the optimized hydrogen-like orbitals will be plotted")
    print("\t The isosurface files of the hydrogen-like orbitals are also genereated in a .ply format")

    for i in range(N):
        plot(optimized_alphas, l, m, n, coefficients[:, i], energy[i], f"orbital_{i}")
    
if __name__ == "__main__":
    main()