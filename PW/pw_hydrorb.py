# -*- 
import numpy as np
import math
import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def main():
    save_psis=True
    plot=False
    npts = 16
    cell_size=200 #bohr 
    tick_size=2
    unitcell = Unitcell(cell_size, npts)
    unitcell.add_atom(cell_size/2, cell_size/2, cell_size/2, 1) # single hydrogen atom
    
    # kinetic energy matrix
    T = 0.5 * np.diag(unitcell.get_pw_k2().flatten())
    
    # build real-space basis functions
    G = unitcell.get_pw_k().reshape(-1,3)
    
    # get positions
    r = unitcell.get_r().reshape(-1,3)
    
    # unitcell volume
    omega = unitcell.get_omega()
    
    # unitcell real-space integration constant
    dV = omega / (npts**3)
    
    # get nuclear potential
    nucpot = unitcell.calculate_vpot().flatten()
    
    # pre-build fields
    ketfields = 1 / np.sqrt(omega) * np.exp(-1j * G @ r.transpose())
    brafields = ketfields.conjugate()
    
    # nuclear potential matrix
    V = np.zeros((npts**3, npts**3), dtype=np.complex128)
    S = np.zeros((npts**3, npts**3), dtype=np.complex128)
    
    # construct jobs for parallel computation
    joblist = []
    for i in range(0, npts**3):
        for j in range(i, npts**3):
            joblist.append((i,j,ketfields[i], nucpot, brafields[j], dV))
    
    for job in tqdm.tqdm(joblist):
        i = job[0]
        j = job[1]
        V[i,j] = np.einsum('i,i,i', brafields[i], nucpot, ketfields[j]) * dV
        S[i,j] = np.einsum('i,i', brafields[i],ketfields[j]) * dV
        if i != j:
            V[j,i] = V[i,j].conjugate()
            S[j,i] = S[i,j].conjugate()
    
    plt.imshow(np.real(S), cmap='PiYG') #plot unitary matrix
    plt.show()
    
    H = T + V
    e,v = np.linalg.eigh(H)
    e += unitcell.calculate_ewald_sum() # needed to correct for divergent G=0 term
    
    # output first 10 energies
    print(e[:30])
    
    if plot==True:
        ticks=np.round(np.linspace(0,cell_size,int(npts/tick_size)),2)
    # plot some results
        fig, ax = plt.subplots(1,2, figsize=(10,20) ,dpi=144)
        ax[0].imshow(np.fft.ifftn(v[:,0].reshape((npts,npts,npts))).real[:,:,npts//2],cmap='PiYG')
        ax[0].set_yticks(np.linspace(0,npts-1,int(npts/tick_size)),ticks)
        ax[0].set_xticks(np.linspace(0,npts-1,int(npts/tick_size)),ticks)
        ax[0].set_title('1s')
        ax[1].imshow(np.fft.ifftn(v[:,2].reshape((npts,npts,npts))).real[:,:,npts//2], cmap='PiYG')
        ax[1].set_yticks(np.linspace(0,npts-1,int(npts/tick_size)),ticks)
        ax[1].set_xticks(np.linspace(0,npts-1,int(npts/tick_size)),ticks)
        ax[1].set_title('2p')
    
    if save_psis==True: #saving unormalized ones, get normlaized in Analysis.py
        for i in range(30):
            np.save(f'PW_psis/{cell_size} angstrum/Psi_{i}',np.fft.ifftn(v[:,i].reshape((npts,npts,npts))).real, allow_pickle=True)
        
    return

class Unitcell:
    """
    Class that encapsulates a cubic unitcell with periodic boundary conditions
    and which can host the nuclei and electrons
    """
    def __init__(self, sz:float, npts:int):
        """Build a periodic system

        Args:
            sz (float): edge size of the cubic unit cell
            npts (int): number of sampling points per Cartesian direction
        """
        
        # size of the cube edges
        self.__sz = sz          
        
        # number of grid points in each cartesian direction
        self.__npts = npts
        
        # unit cell volume
        self.__Omega = sz**3
        
        # unit cell matrix
        self.__unitcell = np.eye(3,3) * sz
        
        # build FFT vectors and store these in the class
        self.__build_fft_vectors()
        
        # create placeholders for atom positions and charges
        self.__atompos = np.zeros((0,3), dtype=np.float64)
        self.__atomchg = np.array([], dtype=np.uint8)
    
    def __str__(self) -> str:
        """string representation of periodic system

        Returns:
            str: string representation
        """
        res = str(self.__unitcell) + '\n'
        for z,a in zip(self.__atomchg, self.__atompos):
            res += '%i  (%6.4f  %6.4f  %6.4f)\n' % (z,a[0], a[1], a[2])
        return res
    
    def add_atom(self, x:float, y:float, z:float, charge:float, unit:str='bohr'):
        """Add an atom to the unit cell

        Args:
            x (float): x-coordinate
            y (float): y-coordinate
            z (float): z-coordinate
            charge (float): charge of the atom
            unit (str, optional): Which length unit, 'bohr' or 'angstrom'. Defaults to 'bohr'.

        Raises:
            Exception: Invalid unit received.
        """
        if unit == 'bohr':
            pos = np.array([x,y,z], dtype=np.float64)
        elif unit == 'angstrom':
            pos = np.array([x,y,z], dtype=np.float64) * 1.8897259886 # angstrom to bohr conversion
        else:
            raise Exception('Unknown unit: %s' % unit)

        # place the atom positions at the end of the Nx3 matrix
        self.__atompos = np.vstack([self.__atompos, pos])
        
        # append the atomic charge
        self.__atomchg = np.append(self.__atomchg, charge)
    
    def get_atom_positions(self) -> np.ndarray:
        """Get atomic positions in real-space

        Returns:
            np.ndarray: atomic positions
        """
        return self.__atompos
    
    def get_atom_charges(self) -> np.ndarray:
        """Get the atomic charges

        Returns:
            np.ndarray: atomic charges
        """
        return self.__atomchg
    
    def get_omega(self) -> float:
        """Get unitcell volume

        Returns:
            float: volume of the unit cell
        """
        return self.__Omega
    
    def get_r(self) -> np.ndarray:
        """Get the sampling vectors in real-space

        Returns:
            np.ndarray: real-space sampling vectors
        """
        return self.__cvec
    
    def get_r_norms(self) -> np.ndarray:
        """Get real-space sampling vector lengths

        Returns:
            np.ndarray: real-space sampling vector lengths
        """
        return np.sqrt(np.einsum('ijkl,ijkl->ijk', self.__cvec, self.__cvec))
    
    def get_pw_k(self) -> np.ndarray:
        """Get the plane wave vectors

        Returns:
            np.ndarray: plane wave vectors
        """
        return self.__kvec
    
    def get_ct(self) -> float:
        """Get the FFT transformation constant from canonical FFT to an FFT using
           a normalized plane wave basis set.

        Returns:
            float: FFT transformation constant
        """
        return np.sqrt(self.__Omega) / self.__npts**3
    
    def get_pw_k2(self) -> np.ndarray:
        """Get the squared length of the plane wave vectors

        Returns:
            np.ndarray: squared length of plane wave vectors
        """
        return self.__k2
    
    def get_npts(self) -> int:
        """Get the number of sampling points per Cartesian direction

        Returns:
            int: number of sampling points per Cartesian direction
        """
        return self.__npts

    def __build_fft_vectors(self):
        """
        Construct the reciprocal space vectors of the plane waves
        """
        # determine grid points in real space
        c = np.linspace(0, self.__sz, self.__npts, endpoint=False)

        # construct real space sampling vectors
        z, y, x = np.meshgrid(c, c, c, indexing='ij')
        
        N = len(c)
        cvec = np.zeros((self.__npts,self.__npts,self.__npts,3))
        cvec[:,:,:,0] = x
        cvec[:,:,:,1] = y
        cvec[:,:,:,2] = z
        
        # calculate plane wave vector coefficients in one dimension
        k = np.fft.fftfreq(self.__npts) * 2.0 * np.pi * (self.__npts / self.__sz)
        
        # construct plane wave vectors
        k3, k2, k1 = np.meshgrid(k, k, k, indexing='ij')
        
        N = len(k)
        kvec = np.zeros((N,N,N,3))
        kvec[:,:,:,0] = k1
        kvec[:,:,:,1] = k2
        kvec[:,:,:,2] = k3
        
        k2 = np.einsum('ijkl,ijkl->ijk', kvec, kvec)
        
        self.__cvec = cvec
        self.__kvec = kvec
        self.__k2 = k2

    def calculate_ewald_sum(self, gcut:float=2, gamma:float=1e-8) -> float:
        """Calculate Ewald sum

        Args:
            gcut (float, optional): Plane wave cut off energy in Ht. Defaults to 2.
            gamma (float, optional): Separation parameter. Defaults to 1e-8.

        Returns:
            float: Ewald sum in Ht
        """
        # establish alpha value for screening Gaussian charges
        alpha = -0.25 * gcut**2 / np.log(gamma)

        # subtract spurious self-interaction
        Eself = np.sqrt(alpha / np.pi) * np.sum(self.__atomchg**2)
        
        # subtract the electroneutrality term using a uniform background charge
        Een = np.pi * np.sum(self.__atomchg)**2 / (2 * alpha * self.__Omega)

        # calculate short-range interaction
        Esr = 0
        amag = np.linalg.norm(self.__unitcell, axis=1) # determine unitcell vector magnitudes
        Nmax = np.rint(np.sqrt(-0.5 * np.log(gamma)) / np.sqrt(alpha) / amag + 1.5)
        T = self.__build_indexed_vectors_excluding_zero(Nmax) @ self.__unitcell

        for ia in range(len(self.__atompos)):
            for ja in range(len(self.__atompos)):
                Rij = self.__atompos[ia] - self.__atompos[ja]       # interatomic distance
                ZiZj = self.__atomchg[ia] * self.__atomchg[ja]      # product of charges
                for t in T:   # loop over all unit cell permutations
                    R = np.linalg.norm(Rij + t)
                    Esr += 0.5 * ZiZj * math.erfc(R * np.sqrt(alpha)) / R
                if ia != ja:  # terms in primary unit cell
                    R = np.linalg.norm(Rij)
                    Esr += 0.5 * ZiZj * math.erfc(R * np.sqrt(alpha)) / R

        # calculate long-range interaction
        Elr = 0
        B = 2 * np.pi * np.linalg.inv(self.__unitcell.T)            # reciprocal lattice vectors
        bm = np.linalg.norm(B, axis=1)                              # vector magnitudes
        s = np.rint(gcut / bm + 1.5)
        G = self.__build_indexed_vectors_excluding_zero(s) @ B      # produce G-vectors
        G2 = np.linalg.norm(G, axis=1)**2                           # calculate G-lengths
        pre = 2 * np.pi / self.__Omega * np.exp(-0.25 * G2 / alpha) / G2

        for ia in range(len(self.__atompos)):
            for ja in range(len(self.__atompos)):
                Rij = self.__atompos[ia] - self.__atompos[ja]
                ZiZj = self.__atomchg[ia] * self.__atomchg[ja]
                GR = np.sum(G * Rij, axis=1)
                Elr += ZiZj * np.sum(pre * np.cos(GR)) # discard imaginary values by using cos
        
        Eewald = Elr + Esr - Eself - Een

        return Eewald

    def __build_indexed_vectors_excluding_zero(self, s):
        """
        Build a set of incrementing vectors from [-s_i,s_i], exclusing the zero-term
        """
        m1 = np.arange(-s[0], s[0] + 1)
        m2 = np.arange(-s[1], s[1] + 1)
        m3 = np.arange(-s[2], s[2] + 1)
        M = np.transpose(np.meshgrid(m1, m2, m3)).reshape(-1, 3)
        return M[~np.all(M == 0, axis=1)] # remove zero-term
    
    def calculate_vpot(self) -> np.ndarray:
        """Construct the nuclear attraction potential

        Returns:
            np.ndarray: nuclear attraction potential in real-space
        """
        # calculate structure factor
        sf = np.exp(1j * self.__kvec @ self.__atompos.T)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            nucpotg = -4.0 * np.pi / self.__k2
            nucpotg[0,0,0] = 0

        # produce the nuclear attraction field      
        vpot = np.fft.fftn(np.einsum('ijk,ijkl,l->ijk', 
                                     nucpotg, 
                                     sf, 
                                     self.__atomchg)) / self.__Omega
        
        return vpot
    
if __name__ == '__main__':
    main()