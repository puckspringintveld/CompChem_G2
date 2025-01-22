Welcome to the code developed for the 6EMAC8, Theoretical and Computational Chemistry Course. This code is developed by Sacha Dekker, David Warrand and Puck Springintveld. 

This repository consists of different folders with codes for the different investogations done in this project. The folders will be discussed seperatly here, with a small introduction and user guide. 

Charge Decomposition

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
