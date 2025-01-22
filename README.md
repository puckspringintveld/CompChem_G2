Welcome to the code developed for the 6EMAC8, Theoretical and Computational Chemistry Course. This code is developed by Sacha Dekker, David Warrand and Puck Springintveld. 

This repository consists of different folders with codes for the different investogations done in this project. The folders will be discussed seperatly here, with a small introduction and user guide. 

Charge Decomposition
The charge decomposition code is written in jupyter notebooks to make the results more accessible. The results are shown without having to re-run the file. The mulliken folder consists of a jupyter notebook where the module PyDFT is used to calculate the density matrix and overlap matrix for both CO and CH4. These matricies are used for mullikin charge decomposition. For the CO the countouplot of the electron density is also shown, as well as the initial analysis for the Bader decomposition where the bader charge volume is created on one axis (Fig: 3.1a in report), this was made using PyDFT as well. The BAder folder consists of the VASP files and the Bader output files, as well as a jupyter notebook with the calculations based on the VASP results.

GTOs, STOs, Real
    To use the code for the STO's and GTO's simply run the main file and follow the requested prompts. 
    To run the code for the "Real" orbitals, which are the analytical solutions to the hydrogen-like wavefunctions simply run the analytical file and follow the prompts.
    To caculate the RMSE and energy deviation create an excel file, which has six rows, the first three being the matching n, l, and m quantum numbers to the created orbitals. The last three rows are RMSE, self-overlap, and Energy Residual percentage.

    The excel file should look as follows: (where for the RMSE, self-overlap and energy residual percentage placeholder values should be used)
        n                           1   2   etc
        l                           0   0   etc
        m                           0   0   etc
        RMSE                        0   0   etc
        Self-overlap                0   0   etc
        Energy Residual percentage  0   0   etc

PW
pw_hydrorb - constructing the hamiltonian, solving eiegen value problem, adding Ewald sum to energies and saving the wavefunctions.
Analysis.py - analysis of the PW wave function results using the save numpy array files which are stored in the PW_psis folder (this is done for several unit cell lenths, folders should be present when running code). Generates analytical solutions using analytical.py, calculates RMSE, plots contour plots (saved in contou plots folder), and makes isosurface files (.ply) which can be processed in Blender (saved in Isosurfaces folder.
analytical - yields analytical hydrogen like plane waves on the same grid as resulting from plane wave analysis. This is used to evaluate the RMSE.
Energy Diagrams - uses the calculated energies for the given wavefunctions (in the correct unit cell) to create the energy diagrams shown in fig: 2.2 and 2.3 in the report. 
The excel file has all the manual analysis done.


