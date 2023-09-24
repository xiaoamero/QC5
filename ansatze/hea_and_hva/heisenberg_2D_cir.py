import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import I,H,X,Z,RX,RY,RZ,CNOT,Rxx,Ryy,Rzz
from .hea import *

def hva_heisenberg_model_2D_unit(size,nthlayer,pbc):
    
    nparams0 = 4*(nthlayer-1)
    p0 = 'p'+str(nparams0)
    p1 = 'p'+str(nparams0+1)
    p2 = 'p'+str(nparams0+2)
    p3 = 'p'+str(nparams0+3)
    cc = Circuit()
    nx,ny = size
    #along X direction
    for ix in range(nx):
        # odd ZZ 
        for iy in range(1,ny-1,2):
            cc += Rzz({p0:1}).on([iy+ix*ny,iy+1+ix*ny])
        if pbc and ny>2:
            cc += Rzz({p0:1}).on([(ix+1)*ny-1,ix*ny])
        # odd YY 
        for iy in range(1,ny-1,2):
            cc += Ryy({p1:1}).on([iy+ix*ny,iy+1+ix*ny])
        if pbc and ny>2:
            cc += Ryy({p1:1}).on([(ix+1)*ny-1,ix*ny])        
        # odd XX
        for iy in range(1,ny-1,2):
            cc += Rxx({p1:1}).on([iy+ix*ny,iy+1+ix*ny])
        if pbc and ny>2:
            cc += Rxx({p1:1}).on([(ix+1)*ny-1,ix*ny])
            
        # even ZZ 
        for iy in range(0,ny-1,2):
            cc += Rzz({p2:1}).on([iy+ix*ny,iy+1+ix*ny])
        # even YY 
        for iy in range(0,ny-1,2):
            cc += Ryy({p3:1}).on([iy+ix*ny,iy+1+ix*ny])
        # even XX 
        for iy in range(0,ny-1,2):
            cc += Rxx({p3:1}).on([iy+ix*ny,iy+1+ix*ny])
            
    # along Y direction      
    for iy in range(ny):
        # odd ZZ 
        for ix in range(1,nx-1,2):
            cc += Rzz({p0:1}).on([iy+ix*ny,iy+(ix+1)*ny])
        if pbc and nx>2:
            cc += Rzz({p0:1}).on([(nx-1)*ny+iy,iy])
        # odd YY
        for ix in range(1,nx-1,2):
            cc += Ryy({p1:1}).on([iy+ix*ny,iy+(ix+1)*ny])
        if pbc and nx>2:
            cc += Ryy({p1:1}).on([(nx-1)*ny+iy,iy])
        # odd XX
        for ix in range(1,nx-1,2):
            cc += Rxx({p1:1}).on([iy+ix*ny,iy+(ix+1)*ny])
        if pbc and nx>2:
            cc += Rxx({p1:1}).on([(nx-1)*ny+iy,iy])

        # even ZZ 
        for ix in range(0,nx-1,2):
            cc += Rzz({p2:1}).on([iy+ix*ny,iy+(ix+1)*ny])
        # even YY 
        for ix in range(0,nx-1,2):
            cc += Ryy({p3:1}).on([iy+ix*ny,iy+(ix+1)*ny])
        # even XX
        for ix in range(0,nx-1,2):
            cc += Rxx({p3:1}).on([iy+ix*ny,iy+(ix+1)*ny])
    return cc

def get_hva_heisenberg_model_2D(size,nlayer,pbc):
    nqubits = size[0]*size[1]
    cir = Circuit()
    for nth in range(1,nlayer+1):
        unit = hva_heisenberg_model_2D_unit(size, nth, pbc)
        cir += unit
    return cir

############################################
#            HEA 2D heisenberg-model ANSATZ
#            nx*ny
############################################
def hea_fSim_linear_2D_heisneberg(size,nlayer):
    nx,ny = size
    ansatz = Circuit()
    for nth in range(1,nlayer+1):
        for ix in range(nx):
            if ny>=2:
                qubitlst = [i for i in range(ix*ny,(ix+1)*ny)]
                ansatz += hea_fSim_linear_unit(qubitlst,nth,'x')
        ansatz.barrier()
        for iy in range(ny):
            if nx>=2:
                qubitlst = [i*ny+iy for i in range(nx)]
                ansatz += hea_fSim_linear_unit(qubitlst,nth,'y')
        ansatz.barrier()
    return ansatz

def hea_fSim_brickwall_2D_heisneberg(size,nlayer):
    nx,ny = size
    ansatz = Circuit()
    for nth in range(1,nlayer+1):
        for ix in range(nx):
            if ny>=2:
                qubitlst = [i for i in range(ix*ny,(ix+1)*ny)]
                ansatz += hea_fSim_brickwall_unit(qubitlst,nth,'x')
        ansatz.barrier()
        for iy in range(ny):
            if nx>=2:
                qubitlst = [i*ny+iy for i in range(nx)]
                ansatz += hea_fSim_brickwall_unit(qubitlst,nth,'y')
        ansatz.barrier()
    return ansatz

def hea_ASWAP_2D_heisneberg(size,nlayer):
    nx,ny = size
    ansatz = Circuit()
    for nth in range(1,nlayer+1):
        for ix in range(nx):
            if ny>=2:
                qubitlst = [i for i in range(ix*ny,(ix+1)*ny)]
                ansatz += hea_ASWAP_unit(qubitlst,nth,'x')
        ansatz.barrier()    
        for iy in range(ny):
            if nx>=2:
                qubitlst = [i*ny+iy for i in range(nx)]
                ansatz += hea_ASWAP_unit(qubitlst,nth,'y')
        ansatz.barrier()
    return ansatz

def hea_ry_cascade_2D_heisneberg(size,nlayer):
    nx,ny = size
    ansatz = Circuit()
    for nth in range(1,nlayer+1):
        for ix in range(nx):
            if ny>=2:
                qubitlst = [i for i in range(ix*ny,(ix+1)*ny)]
                if nth == 1:
                    ansatz += hea_ry_pre(qubitlst,'x')
                ansatz += hea_ry_cascade_unit(qubitlst,nth,'x')
        ansatz.barrier()    
        for iy in range(ny):
            if nx>=2:
                qubitlst = [i*ny+iy for i in range(nx)]
                if nth == 1:
                    ansatz += hea_ry_pre(qubitlst,'y')
                ansatz += hea_ry_cascade_unit(qubitlst,nth,'y')
        ansatz.barrier()
    return ansatz

def pchea_2D_heisneberg(size,nlayer,hea_type):
    gate_orden = hea_type[:5]
    ansatz = Circuit()
    nx,ny = size
    for nth in range(1,nlayer+1):
        for ix in range(nx):
            if ny>=2:
                qubitlst = [i for i in range(ix*ny,(ix+1)*ny)]
                ansatz += pchea_unit(qubitlst,nth,'x',gate_orden)
        ansatz.barrier()
        for iy in range(ny):
            if nx>=2:
                qubitlst = [i*ny+iy for i in range(nx)]
                ansatz += pchea_unit(qubitlst,nth,'y',gate_orden)
        ansatz.barrier()
    return ansatz