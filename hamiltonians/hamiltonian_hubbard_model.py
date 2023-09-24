import numpy as np
from functools import reduce
from mindquantum.core.operators import FermionOperator,QubitOperator
from mindquantum.core.operators.utils import hermitian_conjugated
from pyscf import ao2mo, gto, scf,fci
from .FerOp_QubitOp import (get_qubit_hamiltonian,get_qubit_operator,unitiy_mo_coeff_format,\
                           get_ao2so_h1_h2_int,get_qubit_adder_operator,get_qubit_na_nb_operator)

##############################################
#
#
#
##############################################

def ABorder_qubit_hubbard_model_2D_molmf(size, U, beta0=0.0,beta1=0.0,pbc=False):
    """ SITE basis, and ABABAB order"""
    thop = get_Hubbard_2D_t(size,pbc)
    eri = get_Hubbard_2D_U(size,U)
    nbas = size[0]*size[1]
    
    mol = gto.M()
    mol.verbose = 3
    mol.nelectron = 2*(nbas//2)
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: thop
    mf.get_ovlp = lambda *args: np.eye(nbas)
    mf._eri = ao2mo.restore(8, eri, nbas)
    mf.kernel()
    
    print('spin',mol.spin)
    molFCI = fci.FCI(mf) 
    res = molFCI.kernel() 
    eFCI = res[0]
    print('eFCI',eFCI)
    
    mo_coeff = np.eye(nbas) # mf.mo_coeff
    
    h1e = mf.get_hcore()
    h2e = mf._eri
    h1s,h2s = get_ao2so_h1_h2_int(mo_coeff,h1e,h2e)
    qham = get_qubit_hamiltonian(0.0,h1s,h2s,'jordan_wigner')

    qsp,qsm = get_qubit_adder_operator(np.eye(nbas), mo_coeff,'jordan_wigner')
    qna,qnb = get_qubit_na_nb_operator(nbas,'jordan_wigner')
    print('penalty hamiltonian: beta0-sp-sm',beta0,'beta1-na-nb',beta1)
    penalty_ham = qham + beta0*qsp*qsm + beta1*(qna-nbas//2)**2 + beta1*(qnb-nbas//2)**2
    return penalty_ham

##########################################################
#               Hubbard   model  Hamiltonian             
#              spin-orbital: A - A
#                            |   |
#                            B - B 
#                            |   |
#                            A - A
#                            |   |
#                            B - B
##########################################################

def selforder_qubit_hubbard_model_2D(size, U, beta0=0.0,beta1=0.0,pbc=False):
    """ SITE basis, and AABB order"""
    thop = get_Hubbard_2D_t(size,pbc)
    eri = get_Hubbard_2D_U(size,U)
    nbas = size[0]*size[1]
    
    mol = gto.M()
    mol.verbose = 3
    mol.nelectron = 2*(nbas//2)
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: thop
    mf.get_ovlp = lambda *args: np.eye(nbas)
    mf._eri = ao2mo.restore(8, eri, nbas)
    mf.kernel()
    
    print('spin',mol.spin)
    molFCI = fci.FCI(mf) 
    res = molFCI.kernel() 
    eFCI = res[0]
    print('eFCI',eFCI)
    
    mo_coeff = np.eye(nbas) # mf.mo_coeff
    h1m,h2m = get_RHF_int_h1h2(thop, eri, mo_coeff)
    h1s,h2s = get_SpinOrbital_h1h2_in_AAAABBBBorder(h1m,h2m)
    
    # in self order
    index = get_ix(size)
    h1ss = h1s[np.ix_(index,index)]
    h2ss = h2s[np.ix_(index,index,index,index)]
    qham = get_qubit_hamiltonian(0.0,h1ss,h2ss,'jordan_wigner')

    qsp,qsm = selforder_qsp_qsm(size)
    qna,qnb = selforder_qna_qnb(size)
    
    print('penalty hamiltonian: beta0-sp-sm',beta0,'beta1-na-nb',beta1)
    penalty_ham = qham + beta0*qsp*qsm + beta1*(qna-nbas//2)**2 + beta1*(qnb-nbas//2)**2
    return penalty_ham

def selforder_qna_qnb(size):
    nx,ny = size
    nsite = nx*ny
    nqubits = 2*nsite
    fna = FermionOperator()
    fnb = FermionOperator()
    for idx,i in enumerate(range(nx)):
        ia = 2*i
        ib = 2*i + 1
        ac = (ny*ia,ny*(ia+1))
        bc = (ny*ib,ny*(ib+1))
        for j in range(ac[0],ac[1]):
            fna += FermionOperator(str(j)+'^ '+str(j)+' ')
        for j in range(bc[0],bc[1]):
            fnb += FermionOperator(str(j)+'^ '+str(j)+' ')
    qna = get_qubit_operator(fna,'jordan_wigner')
    qnb = get_qubit_operator(fnb,'jordan_wigner')
    return qna,qnb

def selforder_qsp_qsm(size):
    nx,ny = size
    nsite = nx*ny
    nqubits = 2*nsite
    lsa = []
    lsb = []
    for idx,i in enumerate(range(nx)):
        ia = 2*i
        lsa += np.arange(ny*ia,ny*(ia+1)).tolist()
        ib = 2*i + 1
        lsb += np.arange(ny*ib,ny*(ib+1)).tolist()
    fsp = FermionOperator()
    for j in range(nsite):
        fsp += FermionOperator(str(lsa[j])+'^ '+str(lsb[j])+' ')
    fsm = hermitian_conjugated(fsp)
    qsp = get_qubit_operator(fsp,'jordan_wigner')
    qsm = get_qubit_operator(fsm,'jordan_wigner')
    return qsp,qsm

def get_ix(size):
    """conver AAAA AAAA BBBB BBBB to 
       AAAA
       BBBB
       AAAA
       BBBB
    """
    nx,ny = size
    ls = []
    for idx,i in enumerate(range(nx)):
        ia = 2*i
        ls += np.arange(ny*ia,ny*(ia+1)).tolist()
    for idx,i in enumerate(range(nx)):
        ib = 2*i + 1
        ls += np.arange(ny*ib,ny*(ib+1)).tolist()    
    return np.array(ls)

##########################################################
#              Hubbard   model  Hamiltonian             
#              spin-orbital: AAAABBBB
#              SITE basis
##########################################################

def qubit_hubbard_hamiltonian(size, U, beta0=0.0,beta1=0.0,pbc=False):
    """ SITE basis, and AABB order"""
    thop = get_Hubbard_2D_t(size,pbc)
    eri = get_Hubbard_2D_U(size,U)
    nbas = size[0]*size[1]
    
    mol = gto.M()
    mol.verbose = 3
    mol.nelectron = 2*(nbas//2)
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: thop
    mf.get_ovlp = lambda *args: np.eye(nbas)
    mf._eri = ao2mo.restore(8, eri, nbas)
    mf.kernel()
    
    print('spin',mol.spin)
    molFCI = fci.FCI(mf) 
    res = molFCI.kernel() 
    eFCI = res[0]
    print('eFCI',eFCI)
    
    mo_coeff = np.eye(nbas) # mf.mo_coeff
    h1m,h2m = get_RHF_int_h1h2(thop, eri, mo_coeff)
    h1s,h2s = get_SpinOrbital_h1h2_in_AAAABBBBorder(h1m,h2m)
    qham = get_qubit_hamiltonian(0.0,h1s,h2s,'jordan_wigner')

    qsp,qsm = get_qubit_adder_operator_AAAABBBB(np.eye(nbas), mo_coeff)
    qna,qnb = get_qubit_na_nb_operator_AAAABBBB(nbas)
    print('penalty hamiltonian: beta0-sp-sm',beta0,'beta1-na-nb',beta1)
    penalty_ham = qham + beta0*qsp*qsm + beta1*(qna-nbas//2)**2 + beta1*(qnb-nbas//2)**2
    return penalty_ham

def get_Hubbard_2D_t(size,pbc=False):
    t=1
    nx,ny = size
    nsite = nx*ny
    thop = np.zeros((nsite,nsite))
    # Row:
    for ix in range(nx):
        for iy in range(ny-1):
            thop[iy+ix*ny,iy+1+ix*ny] = -t
            thop[iy+1+ix*ny,iy+ix*ny] = -t
        if pbc and ny>2:
            thop[ix*ny,(ix+1)*ny-1] = -t
            thop[(ix+1)*ny-1,ix*ny] = -t
    # Up:
    for iy in range(ny):
        for ix in range(nx-1):
            thop[iy+ix*ny,iy+(ix+1)*ny] = -t
            thop[iy+(ix+1)*ny,iy+ix*ny] = -t
        if pbc and nx>2:
            thop[iy,iy+(nx-1)*ny] = -t 
            thop[iy+(nx-1)*ny,iy] = -t 
    return thop

def get_Hubbard_2D_U(size,U):
    #t=1, U/t=U
    nx,ny = size
    nbas = nx*ny
    h2e = np.zeros((nbas,nbas,nbas,nbas))
    for i in range(nbas):
        h2e[i,i,i,i] = U
    return h2e

# spatial to spin-orbital integrals
def get_SpinOrbitalERI_h1_in_AAAABBBBorder(h1s):
    nb = h1s.shape[0]
    h1 = np.zeros((2*nb,2*nb))
    h1[:nb,:nb] = h1s # AA
    h1[nb:,nb:] = h1s # BB
    return h1

# chemist notation [ij|kl]
def get_SpinOrbitalERI_h2_in_AAAABBBBorder(h2s):
    nb = h2s.shape[0]
    h2 = np.zeros((2*nb,2*nb,2*nb,2*nb))
    h2[:nb,:nb,:nb,:nb] = h2s # AAAA
    h2[nb:,nb:,nb:,nb:] = h2s # BBBB
    h2[:nb,:nb,nb:,nb:] = h2s # AABB
    h2[nb:,nb:,:nb,:nb] = h2s # BBAA
    return h2 # return h2*0.5 for Qiskit

def get_SpinOrbital_h1h2_in_AAAABBBBorder(h1s,h2s):
    h1 = get_SpinOrbitalERI_h1_in_AAAABBBBorder(h1s)
    h2 = get_SpinOrbitalERI_h2_in_AAAABBBBorder(h2s)
    return h1,h2

# ao -> RHF mo
def get_RHF_int_h1h2(thop, eri, mo_coeff):
    h1 = mo_coeff.T.dot(thop).dot(mo_coeff)
    h2 = eri
    h2 = np.einsum("pqrs,pi->iqrs",h2,mo_coeff)
    h2 = np.einsum("iqrs,qj->ijrs",h2,mo_coeff)
    h2 = np.einsum("ijrs,rk->ijks",h2,mo_coeff)
    h2 = np.einsum("ijks,sl->ijkl",h2,mo_coeff)
    return h1,h2

def get_fermion_ss_operator_AAAABBBB(ovlp,mo_coeff,list_active=[]):   
    # input mo_coeff = mf.mo_coeff
    """ss = 1/2*(s_plus*s_minus + s_minus*s_plus) + s_z**2"""
    mo_coeff = unitiy_mo_coeff_format(mo_coeff)
    s_plus = FermionOperator()
    s_minus = FermionOperator()
    s_z = FermionOperator()    
    ovlpab = reduce(np.dot, (mo_coeff[0].T, ovlp, mo_coeff[1]))
    if list_active:
        ovlpab = ovlpab[np.ix_(list_active,list_active)]
    k = ovlpab.shape[0]
    for p in range(k):
        for q in range(k):
            s_plus += ovlpab[p,q]*FermionOperator(((p,1),(q+k,0)))
    s_minus = hermitian_conjugated(s_plus)
    for p in range(k):
        s_z += 0.5*(FermionOperator(((p,1),(p,0)))-FermionOperator(((p+k,1),(p+k,0))))   
    ss = 1/2*(s_plus*s_minus + s_minus*s_plus) + s_z**2
    ss.compress()
    return ss

def get_qubit_ss_operator_AAAABBBB(ovlp,mo_coeff,fermion_transform='jordan_wigner',list_active=[]):
    """return molecule Qubit ss operator """
    ss = get_fermion_ss_operator_AAAABBBB(ovlp,mo_coeff,list_active)
    qubit_ss_operator = get_qubit_operator(ss,fermion_transform)
    return qubit_ss_operator

def get_fermion_adder_operator_AAAABBBB(ovlp,mo_coeff,list_active=[]):
    """s_plus and s_minus"""
    mo_coeff = unitiy_mo_coeff_format(mo_coeff)
    s_plus = FermionOperator()
    ovlpab = reduce(np.dot, (mo_coeff[0].T, ovlp, mo_coeff[1]))
    if list_active:
        ovlpab = ovlpab[np.ix_(list_active,list_active)]
    nmo = ovlpab.shape[0]
    for p in range(nmo):
        for q in range(nmo):
            s_plus += ovlpab[p,q]*FermionOperator(((p,1),(q+nmo,0)))
    s_minus = hermitian_conjugated(s_plus)
    return s_plus,s_minus

def get_qubit_adder_operator_AAAABBBB(ovlp,mo_coeff,fermion_transform='jordan_wigner',list_active=[]):
    splus,sminus = get_fermion_adder_operator_AAAABBBB(ovlp,mo_coeff,list_active)
    qubit_splus = get_qubit_operator(splus,fermion_transform)
    qubit_sminus = get_qubit_operator(sminus,fermion_transform)
    return qubit_splus,qubit_sminus

def get_fermion_na_nb_operator_AAAABBBB(norb):
    """ nalpha and nbeta"""
    na = FermionOperator()
    nb = FermionOperator() # n_partical
    for p in range(norb):         
        na += FermionOperator(((p,1),(p,0)))
        nb += FermionOperator(((p+norb,1),(p+norb,0)))
    return na,nb

def get_qubit_na_nb_operator_AAAABBBB(norb,fermion_transform='jordan_wigner'):
    """return qubit nalpha nbeta operator"""
    na,nb = get_fermion_na_nb_operator_AAAABBBB(norb)
    qubit_na = get_qubit_operator(na,fermion_transform)
    qubit_nb = get_qubit_operator(nb,fermion_transform)
    return qubit_na,qubit_nb

# 下面的模型是根据公式写的，可以用于验证。
def get_fermion_hubbard_model_2D(size,U,t=1,pbc=False):
    """ 
    spin-orbital: AAAABBBB,
    SITE basis,
    \hat{H} = -t\sum_{<i,j>}\sum_{\sigma}(a^{\dagger}_{i\sigma}a_{j\sigma} + h.c.)
              +U\sum_{i}\hat{n}_{i\alpha}\hat{n}_{i\beta},
    """
    def subterm(label1,label2,nsite):
        alpha = FermionOperator(str(label1)+'^ ')*FermionOperator(str(label2)+' ')
        beta = FermionOperator(str(label1+nsite)+'^ ')*FermionOperator(str(label2+nsite)+' ')
        return alpha+beta
    nx, ny = size
    nsite = nx*ny
    # hopping 
    ham1 = FermionOperator()
    for ix in range(nx):
        for iy in range(ny-1):
            ham1 += subterm(iy+ix*ny,iy+1+ix*ny,nsite)
        if pbc and ny>2:
            ham1 += subterm(ix*ny,(ix+1)*ny-1,nsite)
    for iy in range(ny):
        for ix in range(nx-1):
            ham1 += subterm(iy+ix*ny,iy+(ix+1)*ny,nsite)
        if pbc and nx>2:
            ham1 += subterm(iy,iy+(nx-1)*ny,nsite)
    # on-site
    ham2 = FermionOperator()
    for i in range(nsite):
        aa = FermionOperator(str(i)+'^ '+str(i)+' ')-0.5
        bb = FermionOperator(str(i+nsite)+'^ '+str(i+nsite)+' ')-0.5
        ham2 += aa*bb
    
    ham = (ham1+hermitian_conjugated(ham1))*-t + ham2*U/t
    
    print('Fermion-hubbard-model size:',size,', PBC:',pbc)
    return ham

def get_qubit_hubbard_model_2D(size,U,t=1,pbc=False,fermion_transform='jordan_wigner'):
    fh = get_fermion_hubbard_model_2D(size,U,t,pbc)
    qh = get_qubit_operator(fh, fermion_transform)
    return qh