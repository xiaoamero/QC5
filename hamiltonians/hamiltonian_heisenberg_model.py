import numpy as np
from mindquantum.core.operators import FermionOperator,QubitOperator

##########################################################
#         Anisotropic Heisenberg Hamiltonian             #
##########################################################

def qubit_heisenberg_model_2D(size, J, pbc=True):
    """
    H = -J/2 \sum_{<ij>} sigma_i sigma_j
    
    """
    
    def subterm(label1, label2):
        ex = 'X'+str(label1)+' '+'X'+str(label2)
        ey = 'Y'+str(label1)+' '+'Y'+str(label2)
        ez = 'Z'+str(label1)+' '+'Z'+str(label2)
        opxx = QubitOperator(ex)
        opyy = QubitOperator(ey)
        opzz = QubitOperator(ez)
        return opxx+opyy+opzz
    
    nx, ny = size
    ham = QubitOperator()
    for ix in range(nx):
        for iy in range(ny-1):
            ham += subterm(iy+ix*ny,iy+1+ix*ny)
        if pbc and ny>2:
            #print('yes1',ix*ny,(ix+1)*ny-1)
            ham += subterm(ix*ny,(ix+1)*ny-1)
    for iy in range(ny):
        for ix in range(nx-1):
            ham += subterm(iy+ix*ny,iy+(ix+1)*ny)
        if pbc and nx>2:
            #print('yes2',iy,iy+(nx-1)*ny)
            ham += subterm(iy,iy+(nx-1)*ny)
    print('Hamiltonian size:',size,', PBC:',pbc)
    return ham*-0.5*J

def get_qubit_aniso_ham(nqubits,KJ_ratio,J=-1,cut='Not'):
    """
    ref: Bertels et al., J. Chem. Theory Comput. 18, 11, 6656–6669 (2022)
    \hat{H}_{aniso} = -0.5J\sum_{i}(X_{i}X_{i+1}+Y_{i}Y_{i+1})-0.5K\sum_{i}Z_{i}Z_{i+1},
    K = J*KJratio,
    get_fermion_aniso_ham(nqubits,KJ_ratio,J=-1) after JWT
    """
    K = J*KJ_ratio
    ham = QubitOperator()
    if type(cut)==str:
        site_list = range(nqubits-1)
    elif type(cut) == int:
        site_list = np.arange(0,cut-1).tolist() + np.arange(cut,nqubits-1).tolist()
    for i in site_list:
        ex = 'X'+str(i)+' '+'X'+str(i+1)
        ey = 'Y'+str(i)+' '+'Y'+str(i+1)
        ez = 'Z'+str(i)+' '+'Z'+str(i+1)
        ham += QubitOperator(ex,-0.5*J) + QubitOperator(ey,-0.5*J)\
        + QubitOperator(ez,-0.5*K)
    return ham

def get_fermion_aniso_ham(nqubits,KJ_ratio,J=-1):
    """
    ref: Bertels et al., J. Chem. Theory Comput. 18, 11, 6656–6669 (2022)
    \hat{H}_{aniso} = -J\sum_{i}(a^{\dagger}_{i}a_{i+1}+h.c.) 
                      -K\sum_{i}(2\hat{n}_{i}\hat{n}_{i+1}-\hat{n}_{i}-\hat{n}_{i+1}+0.5),
    K = J*KJratio,
    """
    K = J*KJ_ratio
    ham = FermionOperator()
    for i in range(nqubits-1):
        ej0 = str(i)+"^ "+str(i+1)
        ej1 = str(i+1)+'^ '+str(i)
        ham += FermionOperator(ej0,-1*J)+FermionOperator(ej1,-1*J)
        ek0 = str(i)+'^ '+str(i)
        ek1 = str(i+1)+'^ '+str(i+1)
        ham += FermionOperator(ek0+" "+ek1,-2*K)\
        + FermionOperator(ek0,K) \
        + FermionOperator(ek1,K) \
        + FermionOperator('',-0.5*K)
    return ham

def get_qubit_aniso_ss(nqubits):
    x = QubitOperator()
    y = QubitOperator()
    z = QubitOperator()
    for i in range(nqubits):
        ex = 'X'+str(i)
        ey = 'Y'+str(i)
        ez = 'Z'+str(i)
        x += QubitOperator(ex)
        y += QubitOperator(ey)
        z += QubitOperator(ez)
    ham = 0.25*(x*x +y*y + z*z)
    return ham

def get_qubit_aniso_sz(nqubits):
    z = QubitOperator()
    for i in range(nqubits):
        ez = 'Z'+str(i)
        z += QubitOperator(ez)
    ham = 0.25*z*z
    return ham

def get_qubit_aniso_np(nqubits):
    npartical = QubitOperator()
    for i in range(nqubits):
        qc = 0.5*(QubitOperator('X'+str(i))-1j*QubitOperator('Y'+str(i))) # creation operator
        qa = 0.5*(QubitOperator('X'+str(i))+1j*QubitOperator('Y'+str(i))) # annihition operator
        npartical += qc*qa
    return npartical

def get_qubit_aniso_na_nb(nqubits):
    qna = QubitOperator()
    qnb = QubitOperator()
    for i in range(0,nqubits,2):
        qc = 0.5*(QubitOperator('X'+str(i))-1j*QubitOperator('Y'+str(i))) # creation operator
        qa = 0.5*(QubitOperator('X'+str(i))+1j*QubitOperator('Y'+str(i))) # annihition operator
        qna += qc*qa
    for i in range(1,nqubits,2):
        qc = 0.5*(QubitOperator('X'+str(i))-1j*QubitOperator('Y'+str(i))) # creation operator
        qa = 0.5*(QubitOperator('X'+str(i))+1j*QubitOperator('Y'+str(i))) # annihition operator
        qnb += qc*qa
    return qna,qnb