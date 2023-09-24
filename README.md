# QC5: A Lightweight Python Package for Quantum Computational Chemistry

The main purpose of this package is to validate our ideas about quantum computational chemistry and enable more detailed and flexible exploration with the help of existing packages. The implementation of parameterized quantum circuits and automatic differentiation for energy gradients was performed using [MindQuantum](https://github.com/mindspore-ai/mindquantum.git). Numerical optimization in VQE relies on the methods integrated in [SciPy](https://github.com/scipy/scipy.git) and [NumPy](https://github.com/numpy/numpy.git). The molecular integrals were generated using the [PySCF](https://github.com/pyscf/pyscf.git). The second quantization and fermion-to-qubit transformation were carried out using [OpenFermion](https://github.com/quantumlib/OpenFermion.git).

It consists of three basic components: **ansatze**, **hamiltonians**, and **algorithms**. 

## Ansatze

Parametrized Quantum Circuit (PQC) used to construct quantum circuits that simulate the electronic structure of molecules:

- **Trotterized Unitary Coupled Cluster**: RuCCSD, RuCCGSD1, RuCCGSD2, UuCCSD, UuCCGSD1, UuCCGSD2,
- **Hardware Efficient Ansatz (HEA)**: ry_linear, ry_full, ry_cascade, EfficientSU2 (RyRz), ASWAP, [PCHEA](https://doi.org/10.48550/arXiv.2105.10275)(our work),
- **Hamiltonian Variation Ansatz (HVA)**: hva_hubbard, hva_heisenberg,
- **Adaptive ansatz**: Fermionic-adapt-vqe, Qubit-adapt-vqe.

## Hamiltonians

These hamiltonians can be used to model the behavior of electrons in molecules:

- **Molecular Hamiltonian** $$\hat{H} = \sum_{pq}[p|h|q]a^{\dagger}_ {p}a_ {q}+\sum_{pqrs}\frac{1}{2}[ps|qr]a^{\dagger}_ {p}a^{\dagger}_ {q}a_{r}a_{s}+E_{NN},$$  
$$[p|h|q] = \int dx \phi^{* }_ {p}(x)\hat{h}\phi_ {q}(x),$$
$$[ps|qr] = \int dx_ {1}dx_ {2}\phi^{* }_ {p}(x_ {1})\phi_ {s}(x_ {1})r^{-1}_ {12}\phi_ {q}(x_ {2})\phi_ {r}(x_ {2}),$$  
where $a^{\dagger}_ {p}a_{p}$ are fermionic creation (annihilation) operators. The coefficients $[p|h|q]$ and $[ps|qr]$ are the one-electron and two-electron molecular integrals in Mulliken notations, respectively, which can be computed from [PySCF](https://github.com/pyscf/pyscf.git).  


- **Fermion-hubbard Hamiltonian**
$$\hat{H}=-t\sum_{&lt i,j &gt}\sum_{\sigma}(a_{i\sigma}^{\dagger}a_{j\sigma}+\mathrm{h.c.}) + U\sum_{i}\hat{n}_ {i\alpha} \hat{n}_{i\beta},$$
with hopping amplitudes $t$ between nearest neighbors $i$ and $j$, on-site repulsion energy $U>0$, $\sigma$ reprents spin-up ($\alpha$) and spin-down ($\beta$).  


- **Heisenberg Hamiltonian** $$\hat{H} = -\frac{J}{2}\sum_{<i,j>}\sigma_{i}\cdot\sigma_{j},$$
if $J<0$ couples sites $i$ and $j$ antiferromagnetically while $J>0$ couples them ferromagnetically. 

## Algorithms

Some variational quantum algorithms:  

- **VQE**: The variational quantum eigensolver is a hybrid quantum-classical approach for solving quantum many-body problems.  
- **VQD**: The Variational Quantum Deflation is an extension of the VQE for calculating electronic excited state energies.
- **VQR**: The variational quantum response approach, for computing linear response properties.  
- **GlobalVQE**: A layerwise optimization strategy for HEA/HVA, this guarantees that the optimized energy decreases monotonically with respect to $L$ for our proposed [PCHEA](https://doi.org/10.48550/arXiv.2105.10275).

## Notice

More features will be added to **QC5** in the future.
