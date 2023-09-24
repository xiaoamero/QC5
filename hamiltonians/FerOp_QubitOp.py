""" Useful functions sets"""

import numpy as np
import scipy
from functools import reduce
from pyscf import ao2mo, gto, scf,fci
from mindquantum.algorithm.nisq.chem.transform import Transform
from mindquantum.third_party.interaction_operator import InteractionOperator
from mindquantum.core.operators import FermionOperator,QubitOperator
from mindquantum.core.operators.utils import hermitian_conjugated
from mindquantum.core.operators.polynomial_tensor import PolynomialTensor

# content:
        # Fermion Tools: basis rotation, fermion operators ……
        # Qubit Tools: Transformation, qubit operators ……
### copy from mindquantum 0.6.1

def get_fermion_operator(operator):
    """Convert the tensor (PolynomialTensor) to FermionOperator.

    Args:
        operator (PolynomialTensor): The `PolynomialTensor` you want to convert to `FermionOperator`.

    Returns:
        fermion_operator, An instance of the FermionOperator class.
    """
    fermion_operator = FermionOperator()

    if isinstance(operator, PolynomialTensor):
        for term in operator:
            fermion_operator += FermionOperator(term, operator[term])
        return fermion_operator

    raise TypeError("Unsupported type of oeprator {}".format(operator))


#####################################
#         Fermion Tools             #
#####################################
# basis rotation # 
def unitiy_mo_coeff_format(cm):
    # unity RHF UHF mo_coeff format
    if type(cm)==tuple:
        return cm
    else:
        return (cm,cm)
    
# chemist notation 
## hamiltonian, from mo to so AB-even-odd-order
def get_SpinOrbitalERI_h1(h1s):
    nb = h1s.shape[0]
    h1 = np.zeros((2*nb,2*nb))
    h1[::2,::2] = h1s # AA
    h1[1::2,1::2] = h1s # BB
    return h1

## chemist notation [ij|kl]
def get_SpinOrbitalERI_h2(h2s):
    nb = h2s.shape[0]
    h2 = np.zeros((2*nb,2*nb,2*nb,2*nb))
    h2[::2,::2,::2,::2]     = h2s # AAAA
    h2[1::2,1::2,1::2,1::2] = h2s # BBBB
    h2[::2,::2,1::2,1::2] = h2s # AABB
    h2[1::2,1::2,::2,::2] = h2s # BBAA
    return h2 

def get_mo2so_h1_h2_int(h1s,h2s):
    h1 = get_SpinOrbitalERI_h1(h1s)    
    h2 = get_SpinOrbitalERI_h2(h2s)
    return h1,h2
       
## hamiltonian,  from ao to so 
def get_ao2so_h1_h2_int(mo_coeff,h1e,h2e):
    """AO->spinorbital(ABAB order)"""
    # input mo_coeff = mf.mo_coeff
    mo_coeff = unitiy_mo_coeff_format(mo_coeff)
    nbas = h1e.shape[0]    
    b = np.zeros((nbas,2*nbas))  
    b[:,::2] = mo_coeff[0].copy()
    b[:,1::2] = mo_coeff[1].copy()
    # INT1e:
    hcore = h1e  
    h1 = b.T.dot(hcore).dot(b)
    h1[::2,1::2]=h1[1::2,::2]=0.
    # INT2e:
    mf_eri = h2e  
    h2 = ao2mo.general(mf_eri,(b,b,b,b),compact=False).reshape(2*nbas,2*nbas,2*nbas,2*nbas)
    h2[::2,1::2,:,:]=h2[1::2,::2,:,:]=h2[:,:,::2,1::2]=h2[:,:,1::2,::2]=0.
    return h1,h2

## dipole z, from ao to so 
def get_mol_int_dpl(mol, mo_coeff):
    # input mo_coeff = mf.mo_coeff
    mo_coeff = unitiy_mo_coeff_format(mo_coeff)
    
    # default origin should be zero, shape=(3,nbas,nbas)
    mol.set_common_orig((0,0,0))
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)[2]
    # --
    nbas = ao_dip.shape[0]    
    b = np.zeros((nbas,2*nbas))  
    b[:,::2] = mo_coeff[0].copy()
    b[:,1::2] = mo_coeff[1].copy()
    # INT1e: 
    so_dip = b.T.dot(ao_dip).dot(b)
    so_dip[::2,1::2] = so_dip[1::2,::2]=0.
    return so_dip

def get_ppp_int_dpl(ao_dip,mo_coeff):
    # input mo_coeff = mf.mo_coeff
    mo_coeff = unitiy_mo_coeff_format(mo_coeff)
    #ao_dip = ppp.dipole_int()
    nbas = ao_dip.shape[0]    
    b = np.zeros((nbas,2*nbas))  
    b[:,::2] = mo_coeff[0].copy()
    b[:,1::2] = mo_coeff[1].copy()
    # INT1e: 
    so_dip = b.T.dot(ao_dip).dot(b)
    so_dip[::2,1::2] = so_dip[1::2,::2]=0.
    return so_dip

# Fermion Operators #
def get_fermion_hamiltonian(ecore,h1,h2):  
    """return molecule Fermion hamiltonian """
    h2 = h2.transpose(0,2,3,1) 
    fermion_hamiltonian = InteractionOperator(ecore,h1,0.5*h2)
    fermion_hamiltonian = get_fermion_operator(fermion_hamiltonian)
    return fermion_hamiltonian

def get_fermion_na_nb_operator(norb):
    """ nalpha and nbeta"""
    na = FermionOperator()
    nb = FermionOperator() # n_partical
    for p in range(norb):         
        na += FermionOperator(((2*p,1),(2*p,0)))
        nb += FermionOperator(((2*p+1,1),(2*p+1,0)))
    return na,nb

def get_fermion_npartical_operator(norb):
    na,nb = get_fermion_na_nb_operator(norb)
    return na+nb

def get_fermion_adder_operator(ovlp,mo_coeff,list_active=[]):
    """s_plus and s_minus"""
    mo_coeff = unitiy_mo_coeff_format(mo_coeff)
    s_plus = FermionOperator()
    ovlpab = reduce(np.dot, (mo_coeff[0].T, ovlp, mo_coeff[1]))
    if list_active:
        ovlpab = ovlpab[np.ix_(list_active,list_active)]
    nmo = ovlpab.shape[0]
    for p in range(nmo):
        for q in range(nmo):
            s_plus += ovlpab[p,q]*FermionOperator(((2*p,1),(2*q+1,0)))
    s_minus = hermitian_conjugated(s_plus)
    return s_plus,s_minus

def get_fermion_ss_operator(ovlp,mo_coeff,list_active=[]):   
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
            s_plus += ovlpab[p,q]*FermionOperator(((2*p,1),(2*q+1,0)))
    s_minus = hermitian_conjugated(s_plus)
    for p in range(k):
        s_z += 0.5*(FermionOperator(((2*p,1),(2*p,0)))-FermionOperator(((2*p+1,1),(2*p+1,0))))   
    ss = 1/2*(s_plus*s_minus + s_minus*s_plus) + s_z**2
    ss.compress()
    return ss

def get_fermion_diag_op(idx):
    """a_{p} and a^{\dagger}_{p}""" 
    ap  = FermionOperator(((idx,0),)) 
    apd = hermitian_conjugated(ap)
    return ap, apd

#####################################
#         Qubit Tools               #
#####################################

def ref_in_parityZ2(nqubits0,nelec):
    A = np.triu(np.ones((nqubits0, nqubits0)))
    #print(A)
    aa = np.zeros(nqubits0//2)
    for i in range(nelec[0]):
        aa[i] = 1
    bb = np.zeros(nqubits0//2)
    for i in range(nelec[1]):
        bb[i] = 1
    hf = np.concatenate((aa[::-1],bb[::-1]))
    #
    init = np.dot(A,hf)
    mod2d = init[::-1]%2
    #print(mod2d)
    mod2d = np.delete(mod2d,[nqubits0-1,nqubits0//2-1])
    #print(mod2d)
    hf_occ = np.where(mod2d == 1)[0]
    print(hf_occ)
    return hf_occ.tolist()

def reductionZ1_np(qop_k,qop_v,phasez1,nqubits):
    Z1 = (nqubits-1, 'Z')
    if Z1 in qop_k:
        v = qop_v*phasez1
    else:
        v = qop_v
    k = ()
    for item in qop_k:
        if item[0]<nqubits-1:
            k = k+(item,) 
    return k,v

def reductionZ2_na(qop_k,qop_v,phasez2,nqubits):
    Z2 = (nqubits//2-1, 'Z')
    if Z2 in qop_k:
        v = qop_v*phasez2
    else:
        v = qop_v
    k = ()
    for item in qop_k:
        if item[0] >= nqubits//2:
            k = k+((item[0]-1,item[1]),) 
        elif item[0]<nqubits//2-1:
            k = k+(item,)
    return k,v

def Z2reduction_qop(qop,nelec):
    nqubits = qop.count_qubits()
    
    mod2_np = (nelec[0]+nelec[1])%2
    mod2_na = nelec[0]%2
    phasez1 = (-1)**mod2_np
    phasez2 = (-1)**mod2_na
    
    print('mod2',mod2_np,mod2_na)
    print('phase',phasez1,phasez2)
    
    q2 = QubitOperator()
    for k,v in qop.terms.items():
        k1,v1 = reductionZ1_np(k,v,phasez1,nqubits)
        #print(k1,v1)
        k2,v2 = reductionZ2_na(k1,v1,phasez2,nqubits)
        #print(k2,v2)
        q2 += QubitOperator(k2)*v2
    q2.compress()
    return q2

def get_qubit_operator(fermion_operator,fermion_transform):
    """return molecule qubit operator under corresponding transform """
    if fermion_transform == 'jordan_wigner':
        qubit_operator = Transform(fermion_operator).jordan_wigner()
    elif fermion_transform == 'parity':
        qubit_operator = Transform(fermion_operator).parity()
    elif fermion_transform == 'bravyi_kitaev':
        qubit_operator = Transform(fermion_operator).bravyi_kitaev()
    elif fermion_transform == 'bravyi_kitaev_tree':
        qubit_operator = Transform(fermion_operator).bravyi_kitaev_tree() 
    qubit_operator.compress()
    return qubit_operator   

def get_qubit_hamiltonian(ecore,h1,h2,fermion_transform):  
    """return molecule Qubit hamiltonian """
    fermion_hamiltonian = get_fermion_hamiltonian(ecore,h1,h2)
    qubit_hamiltonian = get_qubit_operator(fermion_hamiltonian,fermion_transform)
    return qubit_hamiltonian

def get_qubit_na_nb_operator(norb,fermion_transform):
    """return qubit nalpha nbeta operator"""
    na,nb = get_fermion_na_nb_operator(norb)
    qubit_na = get_qubit_operator(na,fermion_transform)
    qubit_nb = get_qubit_operator(nb,fermion_transform)
    return qubit_na,qubit_nb

def get_qubit_npartical_operator(norb,fermion_transform):
    """return qubit npartical operator"""
    npartical = get_fermion_npartical_operator(norb)
    qubit_npartical = get_qubit_operator(npartical,fermion_transform)
    return qubit_npartical

def get_qubit_na_nb_preserve_operator(norb,nelec,fermion_transform):
    """return qubit nalpha nbeta preserve operator"""
    na,nb = get_fermion_na_nb_operator(norb)
    qubit_na = get_qubit_operator(na,fermion_transform)-nelec[0]
    qubit_nb = get_qubit_operator(nb,fermion_transform)-nelec[1]
    return qubit_na,qubit_nb

def get_qubit_adder_operator(ovlp,mo_coeff,fermion_transform,list_active=[]):
    splus,sminus = get_fermion_adder_operator(ovlp,mo_coeff,list_active)
    qubit_splus = get_qubit_operator(splus,fermion_transform)
    qubit_sminus = get_qubit_operator(sminus,fermion_transform)
    return qubit_splus,qubit_sminus

def get_qubit_ss_operator(ovlp,mo_coeff,fermion_transform,list_active=[]):
    """return molecule Qubit ss operator """
    ss = get_fermion_ss_operator(ovlp,mo_coeff,list_active)
    qubit_ss_operator = get_qubit_operator(ss,fermion_transform)
    return qubit_ss_operator

# # # # VQR green's funs # # # #
def get_qubit_diag_op(idx,fermion_transform):
    """a_{p} and a^{\dagger}_{p}""" 
    ap,apd = get_fermion_diag_op(idx)
    qubit_ap  = get_qubit_operator(ap,fermion_transform)
    qubit_ap.compress()
    qubit_apd = get_qubit_operator(apd,fermion_transform)
    qubit_apd.compress()
    return qubit_ap, qubit_apd

# # # # VQR azz # # # #
def get_qubit_dipolez(so_zpq,fermion_transform):
    fermion_dipolez = FermionOperator()
    threshold = 1e-10
    for p in range(so_zpq.shape[0]):
        for q in range(so_zpq.shape[0]):
            if abs(so_zpq[p,q]) > threshold:
                fermion_dipolez += so_zpq[p,q]*FermionOperator(((p,1),(q,0)))  
    qubit_dipolez = get_qubit_operator(fermion_dipolez,fermion_transform)
    return qubit_dipolez  

def get_qubit_opVdV(opV):
    opVdV = hermitian_conjugated(opV)*opV
    opVdV.compress()
    return opVdV

def get_qubit_opA(opH,e0,omega,gamma):
    # \hat{A} non-hermian
    opA = opH - QubitOperator('')*(e0 + omega + 1j*gamma)
    opA.compress()
    # \hat{A}^{\dagger}*\hat{A} hermitian
    opAd = hermitian_conjugated(opA)
    opAdA = opAd*opA
    opAdA.compress()
    # opAd_real, opAd_imag
    opAd_imag = gamma
    opAd_real = opH - QubitOperator('')*(e0 + omega)
    opAd_real.compress()
    return opAdA,opA,opAd,opAd_real,opAd_imag

def seperate_real_imag(non_hermitian_qubit_op):
    """non_hermitian_op = real_op + i*imag_op"""
    real_op = QubitOperator()
    imag_op = QubitOperator()
    for k,v in non_hermitian_qubit_op.terms.items():
        if np.abs(np.imag(v))>0:
            imag_op += QubitOperator(k)*(np.imag(v))
    for k,v in non_hermitian_qubit_op.terms.items():
        if np.abs(np.real(v))>0:
            real_op += QubitOperator(k)*(np.real(v))   
    return real_op,imag_op

def get_qubit_opVdA(opV,opA):
    opVdA = hermitian_conjugated(opV)*opA
    opVdA_real,opVdA_imag = seperate_real_imag(opVdA)
    return opVdA,opVdA_real,opVdA_imag
       