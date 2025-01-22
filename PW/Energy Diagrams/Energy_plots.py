# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:22:43 2025

@author: 20211382
"""

import numpy as np
import matplotlib.pyplot as plt

red='#CA0B00FF'
energies=[-0.41483,
          -0.08928,
          -0.08602,
          -0.08602,
          -0.08602,
          -0.03014,
          -0.02999,
          -0.02999,
          -0.02999,
          -0.05320,
          -0.05320,
          -0.03681,
          -0.03681,
          -0.03681,
          -0.02402,
          -0.02402,
          -0.02402,
          -0.02029,
          -0.02029,
          -0.01925,
          -0.01736,
          -0.01736,
          -0.01736,
          -0.01576,
          -0.01569,
          -0.01569,
          -0.01517,
          -0.01517,
          -0.01517,
          -0.01334]

#2
plt.plot(energies[1:5], '_', c=red,markersize=40)
plt.xticks([0,1,2,3],['2s',f'$2p_x$',f'$2p_y$', f'$2p_z$'])
plt.xlim(-1,4)
plt.ylabel('Energy (Ht)')
plt.savefig('energies_2', dpi=300, bbox_inches="tight")  # Ensure high-quality saving
plt.show()

#3
plt.plot(energies[5:14], '_', c=red,markersize=25)
plt.xticks([0,1,2,3,4,5,6,7,8],['3s','$3p_x$','$3p_y$', '$3p_z$','$3d_{z^2}$','$3d_{zx}$','$3d_{yz}$','$3d_{xy}$','$3d_{x^2-y^2}$'])
plt.xlim(-1,9)
plt.ylabel('Energy (Ht)')
plt.savefig('energies_3', dpi=300, bbox_inches="tight")  # Ensure high-quality saving
plt.show()
