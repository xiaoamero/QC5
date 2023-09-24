import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import I,H,X,Z,RX,RY,RZ,CNOT
from mindquantum.core.operators import QubitOperator

from .hf_cir import *
from .hea import *
from .heisenberg_2D_cir import *
from .hubbard_2D_cir import *
from .cir_info import *

def cal_ncnot(nqubits,nlayer,hea_type,size=(0,0),nref=0):
    if nlayer==0:
        return nref
    else:
        if hea_type in ['ry_linear','ASWAP','fSim_linear','fSim_brickwall']:
            return (nqubits-1)*nlayer
        elif hea_type in ['ry_full']:
            return nqubits*(nqubits-1)//2*nlayer
        elif hea_type in ['ry_cascade','XYZ2F','YXY2F']:
            return 2*(nqubits-1)*nlayer
        elif hea_type in ['EfficientSU2']:
            return nqubits*(nqubits-1)//2*nlayer
        
        elif hea_type in ['hva_heisenberg']:
            assert(size!=(0,0))
            return get_hva_heisenberg_pbc_ncnot(size,nlayer,pbc=False)
        elif hea_type in ['hva_heisenberg_pbc']:
            assert(size!=(0,0))
            return get_hva_heisenberg_pbc_ncnot(size,nlayer,pbc=True)
        elif hea_type in ['ry_cascade_2D_heisenberg','XYZ2F_2D_heisenberg']:
            assert(size!=(0,0))
            nx,ny = size
            return 2*(nx-1)*ny*nlayer+2*(ny-1)*nx*nlayer
        elif hea_type in ['ASWAP_2D_heisenberg','fSim_linear_2D_heisenberg','fSim_brickwall_2D_heisenberg']:
            assert(size!=(0,0))
            nx,ny = size
            return (nx-1)*ny*nlayer+(ny-1)*nx*nlayer
        elif hea_type in ['hva_hubbard']:
            nx,ny = size
            nsite = nx*ny
            nqubits = 2*nsite
            num0 = nsite + (ny-1)*nx*2 + (2*(ny-1)+1)*ny*2 # taking exp-i\theta/2(XX+YY) as 1 gate
            return num0*nlayer 
        
        elif hea_type in ['XYZ2F_2D_hubbard']:
            nx,ny = size
            nsite = nx*ny
            num0 = 2*(ny-1)*2*nx + 2*(2*nx-1)*ny
            return num0*nlayer
        elif hea_type in ['ry_cascade_2DX_hubbard','XYZ2F_2DX_hubbard']:
            nx,ny = size
            nsite = nx*ny
            num = 2*(nsite-1)*nlayer*2 + 2*(2-1)*nlayer*nsite
            return num
        elif hea_type in ['fSim_linear_2DX_hubbard','fSim_brickwall_2DX_hubbard']:
            nx,ny = size
            nsite = nx*ny
            return (nsite-1)*nlayer*2+(2-1)*nlayer*nsite
        else:
            print('error hea_type',hea_type)
            exit(1)
            
def cal_ndepth(nqubits,nlayer,hea_type,size=(0,0),nref=0):
    if nlayer==0:
        return nref
    else:
        if hea_type in ['ry_linear']:
            return ry_linear_ndepth(nqubits,nlayer)
        elif hea_type in ['ry_full']:
            return ry_full_ndepth(nqubits,nlayer)
        elif hea_type in ['ry_cascade']:
            return 2*nqubits*nlayer+1
        elif hea_type in ['EfficientSU2']:
            return EfficientSU2_ndepth(nqubits,nlayer)
        elif hea_type in ['XYZ2F','YXY2F']:
            return (4*nqubits+3)*nlayer
        elif hea_type in ['ASWAP','fSim_brickwall']:
            return 2*nlayer
        elif hea_type in ['fSim_linear']:
            return (nqubits-1)*nlayer
        
        elif hea_type in ['hva_heisenberg','hva_heisenberg_pbc']:
            assert(size!=(0,0))
            return hva_heisenberg_ndepth(size,nlayer)
        elif hea_type in ['hva_hubbard']: 
            nx,ny = size
            nsite = nx*ny
            nqubits = 2*nsite
            num0 = 1+2+(2*(ny-1)+1+ny)*ny # taking exp-i\theta/2(XX+YY) as 1 gate
            return num0*nlayer
        
        elif hea_type in ['ry_cascade_2D_heisenberg']:
            assert(size!=(0,0))
            nx,ny = size
            return 2*nx*nlayer+1 + 2*ny*nlayer+1
        elif hea_type in ['XYZ2F_2D_heisenberg']:
            assert(size!=(0,0))
            nx,ny = size
            return (4*nx+3)*nlayer + (4*ny+3)*nlayer
        elif hea_type in ['ASWAP_2D_heisenberg','fSim_brickwall_2D_heisenberg']:
            return 4*nlayer
        elif hea_type in ['fSim_linear_2D_heisenberg']:
            assert(size!=(0,0))
            nx,ny = size
            return (nx+ny-2)*nlayer
        
        
        elif hea_type in ['XYZ2F_2D_hubbard']:
            nx,ny = size
            nsite = nx*ny
            return (4*ny+3)*nlayer+(4*2*nx+3)*nlayer
        elif hea_type in ['ry_cascade_2DX_hubbard']:
            nx,ny = size
            nsite = nx*ny
            return 2*nsite*nlayer+1 + 2*2*nlayer+1
        elif hea_type in ['XYZ2F_2DX_hubbard']:
            nx,ny = size
            nsite = nx*ny
            return (4*nsite+3)*nlayer + (4*2+3)*nlayer
        elif hea_type in ['fSim_brickwall_2DX_hubbard']:
            return brickwall_depth(size,nlayer)
        elif hea_type in ['fSim_linear_2DX_hubbard']:
            nx,ny = size
            nsite = nx*ny
            return (nsite-1)*nlayer + (2-1)*nlayer
        else:
            print('error hea_type',hea_type)
            exit(1)
    
def get_hea_nparams(nqubits,nlayer,hea_type,size=(0,0)):
    
    if nlayer == 0:
        return 0
    else:
        if hea_type in ['hva_heisenberg','hva_heisenberg_pbc']:
            if nqubits//2==2:
                return 2*nlayer #size = (2,2),(1,4),
            else:
                return 4*nlayer
        elif hea_type in ['hva_hubbard']:
            nx,ny = size
            nsite = nx*ny
            na = nsite//2
            if size in [(2,2),(1,4),(1,6),(1,8)]:
                return 3*nlayer + (nsite-na)*na  #size = (2,2),(1,4),
            elif size in [(2,3),(2,4)]:
                return 4*nlayer + (nsite-na)*na
            else:
                return 5*nlayer + (nsite-na)*na
        elif hea_type in ['hva_hubbard_2DY']:
            nx,ny = size
            nsite = nx*ny
            nqubits = 2*nsite
            if size in [(2,2),(1,4),(1,6),(1,8)]:
                return 3*nlayer + (nqubits-nsite)*nsite
            elif size in [(2,3)]:
                return 4*nlayer + (nqubits-nsite)*nsite
            else:
                return 5*nlayer + (nqubits-nsite)*nsite
        
        # hubbard 
        elif hea_type in ['fSim_linear_2D_hubbard']:
            nx,ny = size
            num0 = (ny-1)*2*nx*2 + (2*nx-1)*2*ny
            return num0*nlayer
        elif hea_type in ['XYZ2F_2D_hubbard']:
            nx,ny = size
            num0 = (5*ny-2)*nx*2+(5*2*nx-2)*ny
            return num0*nlayer
        
        elif hea_type in ['fSim_linear_2DX_hubbard','fSim_brickwall_2DX_hubbard']:
            nx,ny = size
            nsite = nx*ny
            nqubits = 2*nsite
            num0 = 3*nqubits-4
            return num0*nlayer
        elif hea_type in ['ry_cascade_2DX_hubbard']:
            nx,ny = size
            nsite = nx*ny
            num = (2*nlayer+1)*nsite*2 + (2*nlayer+1)*2*nsite
            return num
        elif hea_type in ['XYZ2F_2DX_hubbard']:
            nx,ny = size
            nsite = nx*ny
            nqubits = 2*nsite
            num0 = 9*nqubits-4
            return num0*nlayer

        # heisenberg  
        elif hea_type in ['ry_cascade_2D_heisenberg']:
            return nqubits*(2*nlayer+1)*2
        elif hea_type in ['XYZ2F_2D_heisenberg','YXY2F_2D_heisenberg']:
            nx,ny = size
            num0 = (5*nx-2)*ny+(5*ny-2)*nx
            return num0*nlayer
        elif hea_type in ['ASWAP_2D_heisenberg','fSim_linear_2D_heisenberg','fSim_brickwall_2D_heisenberg']:
            nx,ny = size
            num = 2*(nx-1)*nlayer*ny + 2*(ny-1)*nlayer*nx
            return num
        
        # general
        elif hea_type in ['ry_full','ry_linear','Ry full','Ry linear']:
            return nqubits*(nlayer+1)
        elif hea_type in ['ry_cascade','cascade']:
            return nqubits*(2*nlayer+1)
        elif hea_type in ['EfficientSU2','RyRz']:
            return nqubits*(2*nlayer+2)
        elif hea_type in ['ASWAP','fSim_linear','fSim_brickwall']:
            return 2*(nqubits-1)*nlayer
        else:
            return get_pchea_nparams(nqubits,nlayer,hea_type)

def get_hea_ansatz(nqubits,nlayer,hea_type,size=(0,0),nelec=(0,0),ref=''):

    if hea_type in ['hva_heisenberg']: 
        ansatz = get_hva_heisenberg_model_2D(size,nlayer,pbc=False)
    elif hea_type in ['hva_heisenberg_pbc']:
        ansatz = get_hva_heisenberg_model_2D(size,nlayer,pbc=True)
    elif hea_type in ['hva_hubbard']:
        ansatz = get_hva_hubbard_model_2D(size,nlayer,pbc=False) #givens rotation as ref by default
    elif hea_type in ['hva_hubbard_2DY']:
        ansatz = get_hva_hubbard_model_2DY(size,nlayer,pbc=False)
    
    # hubbard
    elif hea_type in ['fSim_linear_2D_hubbard']:
        ansatz = hea_fSim_linear_2D_hubbard(size,nlayer)
    elif hea_type in ['XYZ2F_2D_hubbard']:
        ansatz = pchea_2D_hubbard(size,nlayer,hea_type)

    elif hea_type in ['ry_cascade_2DX_hubbard']:
        ansatz = hea_ry_cascade_2DX_hubbard(size,nlayer)
    elif hea_type in ['fSim_linear_2DX_hubbard']:
        ansatz = hea_fSim_linear_2DX_hubbard(size,nlayer)
    elif hea_type in ['fSim_brickwall_2DX_hubbard']:
        ansatz = hea_fSim_brickwall_2DX_hubbard(size,nlayer)
    elif hea_type in ['XYZ2F_2DX_hubbard']:
        ansatz = pchea_2DX_hubbard(size,nlayer,hea_type)
        
    # heisenberg 
    elif hea_type in ['ry_cascade_2D_heisenberg']:
        ansatz = hea_ry_cascade_2D_heisneberg(size,nlayer)
    elif hea_type in ['XYZ2F_2D_heisenberg','YXY2F_2D_heisenberg']:
        ansatz = pchea_2D_heisneberg(size,nlayer,hea_type)
    elif hea_type in ['ASWAP_2D_heisenberg']:
        ansatz = hea_ASWAP_2D_heisneberg(size,nlayer)
    elif hea_type in ['fSim_linear_2D_heisenberg']:
        ansatz = hea_fSim_linear_2D_heisneberg(size,nlayer)
    elif hea_type in ['fSim_brickwall_2D_heisenberg']:
        ansatz = hea_fSim_brickwall_2D_heisneberg(size,nlayer)
    
    # general or 1D like model
    elif hea_type in ['ry_linear','Ry linear']:
        ansatz = hea_ry_linear(nqubits,nlayer)
    elif hea_type in ['ry_full','Ry full']:
        ansatz = hea_ry_full(nqubits,nlayer)
    elif hea_type in ['ry_cascade','cascade']:
        ansatz = hea_ry_cascade(nqubits,nlayer)
    elif hea_type in ['EfficientSU2','RyRz']:
        ansatz = hea_RyRz(nqubits,nlayer)
    elif hea_type in ['ASWAP']:
        ansatz = hea_ASWAP(nqubits,nlayer)
    elif hea_type in ['fSim_linear']:
        ansatz = hea_fSim_linear(nqubits,nlayer)
    elif hea_type in ['fSim_brickwall']:
        ansatz = hea_fSim_brickwall(nqubits,nlayer)
    else:
        ansatz = pchea(nqubits,nlayer,hea_type)
    
    hf = hf_input(nqubits, nelec, ref)
    if nlayer == 0:
        return hf
    else:
        return hf+ansatz
    
def get_hea_psi(amplitudes,nqubits,nlayer,hea_type,size=(0,0),nelec=(0,0),ref=''):
    ansatz = get_hea_ansatz(nqubits,nlayer,hea_type,size,nelec,ref)
    pr = dict(zip(ansatz.params_name,amplitudes[:len(ansatz.params_name)]))
    psi = ansatz.get_qs(pr=pr)
    return psi

def analyze_psi(psi,thres=0.01):
    op = 0
    for idx in range(len(psi)):
        ratio = np.abs(psi[idx])**2
        if ratio >= thres:
            op += ratio
            print('{} {} {: .6f} {: .6f}'.format(idx,bin(idx),psi[idx],ratio))
    print('sum:',op)
    return None