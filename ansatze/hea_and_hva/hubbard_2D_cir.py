import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import I,H,X,Z,RX,RY,RZ,CNOT,Rxx,Ryy,Rzz
from .hea import *

def Givens_rotation_ref_2DY(size):
    """ 
    gate arrangement ref: Nature Communications| (2022) 13:5743 FIG 1d
    
    """
    nsite = size[0]*size[1]
    nqubits = 2*nsite
    nelec = (nsite//2,nsite//2)
    occlst = [i for i in range(nsite)]
    
    cir = Circuit([X.on(i) for i in occlst])
    for j in range(nqubits//2):
        for i in range(nqubits//2-1+j,-1+j,-1):
            cir += Givens_rotation_gate(i,i+1,'l'+str(j)+'g'+str(i))
    return cir

def hva_hubbard_model_2DY_unit(size,nthlayer,pbc):
    """ AAA
        BBB
        AAA
        BBB 
    """
    p0 = 'l'+str(nthlayer)+'p0'
    p1 = 'l'+str(nthlayer)+'p1'
    p2 = 'l'+str(nthlayer)+'p2'
    p3 = 'l'+str(nthlayer)+'p3'
    p4 = 'l'+str(nthlayer)+'p4'
    
    cc = Circuit()
    nx,ny = size
    nsite = nx*ny
    nqubits = 2*nsite
    # on-site
    for i in range(nx): 
        for j in range(ny):
            cc += Rzz({p0:1}).on([j+i*nsite,j+ny+i*nsite])
    cc.barrier()
    # hopping AA
    # along X direction
    for ix in range(nx):
        ls = [iy+ix*2*ny for iy in range(ny)]
        # even    
        for iy in range(0,ny-1,2):
            cc += hopping_X(ls[iy],ls[iy+1],p1)
        # odd
        for iy in range(1,ny-1,2):
            cc += hopping_X(ls[iy],ls[iy+1],p2)
        if pbc and ny>2:
            cc += hopping_X(ls[0],ls[-1],p2)
    cc.barrier()
    # along Y direction      
    for iy in range(ny):
        ls = [iy+2*ix*ny for ix in range(nx)]
        # even 
        for ix in range(0,nx-1,2):
            cc += hopping_Y(ls[ix],ls[ix+1],p3)
        # odd 
        for ix in range(1,nx-1,2):
            cc += hopping_Y(ls[ix],ls[ix+1],p4)
        if pbc and ny>2:
            cc += hopping_Y(ls[0],ls[-1],p4)
    cc.barrier()
    # hopping BB
    # along X direction
    for ix in range(nx):
        ls = [iy+(ix*2+1)*ny for iy in range(ny)]
        # even    
        for iy in range(0,ny-1,2):
            cc += hopping_X(ls[iy],ls[iy+1],p1)
        # odd
        for iy in range(1,ny-1,2):
            cc += hopping_X(ls[iy],ls[iy+1],p2)
        if pbc and ny>2:
            cc += hopping_X(ls[0],ls[-1],p2)
    cc.barrier()
    # along Y direction      
    for iy in range(ny):
        ls = [iy+(2*ix+1)*ny for ix in range(nx)]
        # even 
        for ix in range(0,nx-1,2):
            cc += hopping_Y(ls[ix],ls[ix+1],p3)
        # odd 
        for ix in range(1,nx-1,2):
            cc += hopping_Y(ls[ix],ls[ix+1],p4)
        if pbc and ny>2:
            cc += hopping_Y(ls[0],ls[-1],p4)
    cc.barrier()
    return cc

def get_hva_hubbard_model_2DY(size,nlayer,pbc=False):
    nsite = size[0]*size[1]
    nqubits = 2*nsite
    cir = Givens_rotation_ref_2DY(size) #hf_input(nqubits,(0,0),ref)
    for nth in range(1,nlayer+1):
        unit = hva_hubbard_model_2DY_unit(size, nth, pbc)
        cir += unit
    return cir


##############################################################
def Givens_rotation_gate(q1,q2,pname):
    """
    G = e^i*phi/2*(X_{i}Y_{i+1}-X_{i+1}Y_{i})
      = np.array([[1,0,0,0],
                 [0,cos(phi),-sin(phi),0],
                 [0,sin(phi), cos(phi),0],
                 [0,0,0,1]])
    gate decompose ref: PHYSICAL REVIEW RESEARCH 4, 023190 (2022) FIG 12
    """
    cir  = Circuit()
    cir += H.on(q1)
    cir += X.on(q2,q1)
    cir += RY({pname:-0.5}).on(q1)
    cir += RY({pname:-0.5}).on(q2)
    cir += X.on(q2,q1)
    cir += H.on(q1)
    return cir

def Givens_rotation_ref(size):
    """ 
    gate arrangement ref: Nature Communications| (2022) 13:5743 FIG 1d
    
    """
    nsite = size[0]*size[1]
    nqubits = 2*nsite
    nelec = (nsite//2,nsite//2)
    occlst = [i for i in range(nelec[0])]+[j+nsite for j in range(nelec[1])]
    
    cir = Circuit([X.on(i) for i in occlst])
    for j in range(nsite//2):
        for i in range(nsite//2-1+j,-1+j,-1):
            cir += Givens_rotation_gate(i,i+1,'l'+str(j)+'g'+str(i))
            cir += Givens_rotation_gate(i+nsite,i+1+nsite,'l'+str(j)+'g'+str(i))
    return cir

def hopping_X(q1,q2,param):
    co = Circuit()
    co += Rxx({param:1}).on([q1,q2]) #AA
    co += Ryy({param:1}).on([q1,q2]) #AA 
    return co

def hopping_Y(q1,q2,param):
    assert(q1+1<q2)
    co = Circuit()
    for qi in range(q1+1,q2):
        co += Z.on(q2,qi)
    co += Rxx({param:1}).on([q1,q2]) #AA
    co += Ryy({param:1}).on([q1,q2]) #AA 
    for qi in range(q1+1,q2)[::-1]:
        co += Z.on(q2,qi)
    return co

def hva_hubbard_model_2D_unit(size,nthlayer,pbc):
    """ AAAABBBB order """
    p0 = 'l'+str(nthlayer)+'p0'
    p1 = 'l'+str(nthlayer)+'p1'
    p2 = 'l'+str(nthlayer)+'p2'
    p3 = 'l'+str(nthlayer)+'p3'
    p4 = 'l'+str(nthlayer)+'p4'
    
    cc = Circuit()
    nx,ny = size
    nsite = nx*ny
    nqubits = 2*nsite
    # on-site
    for i in range(nsite):
        cc += Rzz({p0:1}).on([i,i+nsite])
    cc.barrier()
    # hopping AA
    # along X direction
    for ix in range(nx):
        ls = [iy+ix*ny for iy in range(ny)]
        # even    
        for iy in range(0,ny-1,2):
            cc += hopping_X(ls[iy],ls[iy+1],p1)
        # odd
        for iy in range(1,ny-1,2):
            cc += hopping_X(ls[iy],ls[iy+1],p2)
        if pbc and ny>2:
            cc += hopping_X(ls[0],ls[-1],p2)
    #cc.barrier()
    # along Y direction      
    for iy in range(ny):
        ls = [iy+ix*ny for ix in range(nx)]
        # even 
        for ix in range(0,nx-1,2):
            cc += hopping_Y(ls[ix],ls[ix+1],p3)
        # odd 
        for ix in range(1,nx-1,2):
            cc += hopping_Y(ls[ix],ls[ix+1],p4)
        if pbc and ny>2:
            cc += hopping_Y(ls[0],ls[-1],p4)
    #cc.barrier()
    # hopping BB
    # along X direction
    for ix in range(nx):
        ls = [iy+ix*ny+nsite for iy in range(ny)]
        # even    
        for iy in range(0,ny-1,2):
            cc += hopping_X(ls[iy],ls[iy+1],p1)
        # odd
        for iy in range(1,ny-1,2):
            cc += hopping_X(ls[iy],ls[iy+1],p2)
        if pbc and ny>2:
            cc += hopping_X(ls[0],ls[-1],p2)
    #cc.barrier()
    # along Y direction      
    for iy in range(ny):
        ls = [iy+ix*ny+nsite for ix in range(nx)]
        # even 
        for ix in range(0,nx-1,2):
            cc += hopping_Y(ls[ix],ls[ix+1],p3)
        # odd 
        for ix in range(1,nx-1,2):
            cc += hopping_Y(ls[ix],ls[ix+1],p4)
        if pbc and ny>2:
            cc += hopping_Y(ls[0],ls[-1],p4)
    #cc.barrier()
    return cc

def get_hva_hubbard_model_2D(size,nlayer,pbc=False):
    nsite = size[0]*size[1]
    nqubits = 2*nsite
    cir = Givens_rotation_ref(size) #hf_input(nqubits,(0,0),ref)
    for nth in range(1,nlayer+1):
        unit = hva_hubbard_model_2D_unit(size, nth, pbc)
        cir += unit
    return cir

# def hopping_AB_alongX(q1,q2,param,nsite):
#     co = Circuit()
#     co += Rxx({param:1}).on([q1,q2]) #AA
#     co += Ryy({param:1}).on([q1,q2]) #AA 
#     co += Rxx({param:1}).on([q1+nsite,q2+nsite]) #BB
#     co += Ryy({param:1}).on([q1+nsite,q2+nsite]) #BB
#     return co
# 
# def hopping_AB_alongY(q1,q2,param,nsite):
#     assert(q1+1<q2)
#     co = Circuit()
#     for qi in range(q1+1,q2):
#         co += Z.on(q2,qi)
#     co += Rxx({param:1}).on([q1,q2]) #AA
#     co += Ryy({param:1}).on([q1,q2]) #AA 
#     for qi in range(q1+1,q2)[::-1]:
#         co += Z.on(q2,qi)
#     
#     for qi in range(q1+1+nsite,q2+nsite):
#         co += Z.on(q2+nsite,qi)
#     co += Rxx({param:1}).on([q1+nsite,q2+nsite]) #BB
#     co += Ryy({param:1}).on([q1+nsite,q2+nsite]) #BB
#     for qi in range(q1+1+nsite,q2+nsite)[::-1]:
#         co += Z.on(q2+nsite,qi)
#     return co
# 
# def hva_hubbard_model_2D_unit(size,nthlayer,pbc):
#     """ AAAABBBB order """
#     p0 = 'l'+str(nthlayer)+'p0'
#     p1 = 'l'+str(nthlayer)+'p1'
#     p2 = 'l'+str(nthlayer)+'p2'
#     p3 = 'l'+str(nthlayer)+'p3'
#     p4 = 'l'+str(nthlayer)+'p4'
#     
#     cc = Circuit()
#     nx,ny = size
#     nsite = nx*ny
#     nqubits = 2*nsite
#     # on-site
#     for i in range(nsite):
#         cc += Rzz({p0:1}).on([i,i+nsite])
#     cc.barrier()
#     # hopping 
#     # along X direction
#     for ix in range(nx):
#         # even    
#         for iy in range(0,ny-1,2):
#             cc += hopping_AB_alongX(iy+ix*ny,iy+1+ix*ny,p1,nsite)
#         # odd
#         for iy in range(1,ny-1,2):
#             cc += hopping_AB_alongX(iy+ix*ny,iy+1+ix*ny,p2,nsite)
#         if pbc and ny>2:
#             cc += hopping_AB_alongX((ix+1)*ny-1,ix*ny,p2,nsite)
#     cc.barrier()
#     # along Y direction      
#     for iy in range(ny):     
#         # even 
#         for ix in range(0,nx-1,2):
#             cc += hopping_AB_alongY(iy+ix*ny,iy+(ix+1)*ny,p3,nsite)
#         # odd 
#         for ix in range(1,nx-1,2):
#             cc += hopping_AB_alongY(iy+ix*ny,iy+(ix+1)*ny,p4,nsite)
#         if pbc and ny>2:
#             cc += hopping_AB_alongY((nx-1)*ny+iy,iy,p4,nsite)
#     cc.barrier()
#     return cc
# 
# def get_hva_hubbard_model_2D(size,nlayer,pbc=True):
#     nsite = size[0]*size[1]
#     nqubits = 2*nsite
#     cir = Givens_rotation_ref(size) #hf_input(nqubits,(0,0),ref)
#     for nth in range(1,nlayer+1):
#         unit = hva_hubbard_model_2D_unit(size, nth, pbc)
#         cir += unit
#     return cir

############################################
#            HEA 2D hubbard-model ANSATZ
#            AAA
#            BBB
#　　　　　　 AAA
#            BBB
############################################

def hea_fSim_linear_2D_hubbard(size,nlayer):
    nx,ny = size
    nsite = nx*ny
    nqubits = 2*nsite
    ansatz = Circuit()
    for nth in range(1,nlayer+1):
        for i in range(2*nx):
            qubitlst = np.arange(ny*i,ny*(i+1),1).tolist()
            ansatz += hea_fSim_linear_unit(qubitlst,nth,'x')
        for j in range(ny):
            qubitlst = (np.arange(0,nqubits,ny)+j).tolist()
            ansatz += hea_fSim_linear_unit(qubitlst,nth,'y')  
        ansatz.barrier()
    return ansatz

def pchea_2D_hubbard(size,nlayer,hea_type):
    """ 
    hea_type:'XYZ2F_2D', gate_orden:'XYZ2F'
    
    """
    nx,ny = size
    nsite =  nx*ny
    nqubits = 2*nsite
    gate_orden = hea_type[:5]
    ansatz = Circuit()
    for nth in range(1,nlayer+1):
        for i in range(2*nx):
            qubitlst = np.arange(ny*i,ny*(i+1),1).tolist()
            ansatz += pchea_unit(qubitlst,nth,'x',gate_orden)
        for j in range(ny):
            qubitlst = (np.arange(0,nqubits,ny)+j).tolist()
            ansatz += pchea_unit(qubitlst,nth,'y',gate_orden)  
        ansatz.barrier()
    return ansatz

############################################
#            HEA 2D hubbard-model ANSATZ
#            AAAAAA
#            BBBBBB
############################################

def hea_ry_cascade_2DX_hubbard(size,nlayer):
    """ 
    AAA BBB
    AAA BBB
    """
    nx,ny = size
    nsite =  nx*ny
    nqubits = 2*nsite
    ansatz = Circuit()
    for nth in range(1,nlayer+1):
        for j in range(2):
            qubitlst = [i+j*nsite for i in range(nsite)]
            if nth==1:
                ansatz += hea_ry_pre(qubitlst,'x')
            ansatz += hea_ry_cascade_unit(qubitlst,nth,'x')
        ansatz.barrier()
        for j in range(nsite):
            qubitlst = [j,j+nsite]
            if nth==1:
                ansatz += hea_ry_pre(qubitlst,'y')
            ansatz += hea_ry_cascade_unit(qubitlst,nth,'y')
        ansatz.barrier()
    return ansatz 

def hea_fSim_brickwall_2DX_hubbard(size,nlayer):
    """ 
    AAA BBB
    AAA BBB
    """
    nx,ny = size
    nsite =  nx*ny
    nqubits = 2*nsite
    ansatz = Circuit()
    for nth in range(1,nlayer+1):
        for j in range(2):
            qubitlst = [i+j*nsite for i in range(nsite)]
            ansatz += hea_fSim_brickwall_unit(qubitlst,nth,'x')
        ansatz.barrier()
        for j in range(nsite):
            qubitlst = [j,j+nsite]
            ansatz += hea_fSim_brickwall_unit(qubitlst,nth,'y')
        ansatz.barrier()
    return ansatz 

def hea_fSim_linear_2DX_hubbard(size,nlayer):
    """ 
    AAA BBB
    AAA BBB
    """
    nx,ny = size
    nsite =  nx*ny
    nqubits = 2*nsite
    ansatz = Circuit()
    for nth in range(1,nlayer+1):
        for j in range(2):
            qubitlst = [i+j*nsite for i in range(nsite)]
            ansatz += hea_fSim_linear_unit(qubitlst,nth,'x')
        ansatz.barrier()
        for j in range(nsite):
            qubitlst = [j,j+nsite]
            ansatz += hea_fSim_linear_unit(qubitlst,nth,'y')
        ansatz.barrier()
    return ansatz 

def pchea_2DX_hubbard(size,nlayer,hea_type):
    """ 
    AAA AAA
    BBB BBB
    """
    nx,ny = size
    nsite =  nx*ny
    nqubits = 2*nsite
    gate_orden = hea_type[:5]
    ansatz = Circuit()
    for nth in range(1,nlayer+1):
        for j in range(2):
            qubitlst = [i+j*nsite for i in range(nsite)]
            ansatz += pchea_unit(qubitlst,nth,'x',gate_orden)
        ansatz.barrier()
        for j in range(nsite):
            qubitlst = [j,j+nsite]
            ansatz += pchea_unit(qubitlst,nth,'y',gate_orden)  
        ansatz.barrier()
    return ansatz   