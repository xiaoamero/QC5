from scipy.special import binom
from .FerOp_QubitOp import *

def CIanalyze_sp(norb,nelec,eFCI,vFCI):  
    # check input pyscf or scipy type
    na,nb = nelec
    nas = int(binom(norb,na))
    nbs = int(binom(norb,nb))
    FCIshape = (nas,nbs)
    if vFCI.shape != FCIshape:
        vFCI = vFCI.reshape(FCIshape)
    # spin
    spin = fci.spin_op.spin_square(vFCI, norb, nelec)[1]   #[0] S^2  
    print("*This state E = {: .6f} a.u. 2S+1={: .2f}".format(eFCI, spin)) 
    # determinant
    check_normalize = CIdeterminant(norb,nelec,vFCI) 
    return None

def CIdeterminant(norb,nelec,vFCI):
    nalpha,nbeta = nelec
    occslst1 = fci.cistring._gen_occslst(range(norb),nalpha)
    occslst2 = fci.cistring._gen_occslst(range(norb),nbeta)
    check_normalize = 0
    for i,occsa in enumerate(occslst1):
        for j,occsb in enumerate(occslst2):
            check_normalize += np.square(vFCI[i,j])
            if abs(vFCI[i,j])>0.1:
                print('det_A_&_B',occsa,occsb, "{: .6f} {:.4f}".\
                      format(vFCI[i,j], np.square(vFCI[i,j])))     
    #            
    if np.isclose(check_normalize,1)==False:
        print('Normalize failure,check_normalize =',check_normalize)   
    return check_normalize

def genAn(atom:str,n:int,Re:float)->list:
    return [[atom, 0., 0., i*Re] for i in range(n)]

def qubit_molecular_hamiltonian(geom,beta0=0,beta1=0,basis='RHF',spin_orbital='ABAB',
                                fermion_transform = 'jordan_wigner',
                                set_basis='sto3g',set_charge=0,set_spin=0):
    """
    geom = [['Li', 0.0, 0.0, 0.0],['H', 0.0, 0.0, 2.0]]
    """
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = set_basis
    mol.charge = set_charge
    mol.spin = set_spin
    mol.build()  
    
    # RHF
    mf = scf.RHF(mol)
    eRHF = mf.kernel()
    print('eRHF',eRHF)

    nelec = mol.nelec
    mo_coeff = mf.mo_coeff 
    norb = mo_coeff.shape[1]
    
    # FCI 
    molFCI = fci.FCI(mf)
    res = molFCI.kernel(nroots=10)
    eFCI = res[0][0]
    vFCI = res[1][0]
    print('eFCI',eFCI)
    CIanalyze_sp(norb,nelec,eFCI,vFCI)
    
    # classical information nelec and integrals
    ecore = mol.energy_nuc()
    h1e = mf.get_hcore()
    h2e = mf._eri    
    ovlp = mf.get_ovlp()
    
    if basis == 'SITE':
        oao = orth.orth_ao(mol, method='meta_lowdin', pre_orth_ao='ANO', s=None)
        mo_coeff = oao
    h1,h2 = get_ao2so_h1_h2_int(mo_coeff,h1e,h2e)
    qham = get_qubit_hamiltonian(ecore,h1,h2,fermion_transform)
    qna,qnb = get_qubit_na_nb_operator(norb,fermion_transform)
    qsp,qsm = get_qubit_adder_operator(ovlp,mo_coeff,fermion_transform)
    
    print('penalty hamiltonian: beta0-sp-sm',beta0,'beta1-na-nb',beta1)
    penalty_ham = qham + beta0*qsp*qsm + beta1*(qna-nelec[0])**2 + beta1*(qnb-nelec[1])**2
    return penalty_ham