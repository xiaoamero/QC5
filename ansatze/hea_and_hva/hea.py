import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import I,H,X,Z,RX,RY,RZ,UnivMathGate,FSim,CNOT
from mindquantum.core.gates import Rxx,Ryy,Rzz

############################################
#
#                  Classical HEA
#
############################################

def hea_ry_pre(qubitlst,xy):
    pre = Circuit()
    for j in qubitlst:
        pre += RY('l0'+xy+'0p'+str(j)).on(j)
    return pre

# ry_full
def hea_ry_full_unit(qubitlst,nthlayer,xy):
    encoder = Circuit()
    for j in range(len(qubitlst)-1):
        for k in range(j+1,len(qubitlst)):
            encoder += X.on(qubitlst[k], qubitlst[j])
    for j in qubitlst:
        encoder += RY('l'+str(nthlayer)+xy+'1p'+str(j)).on(j)     
    return encoder
                                
def hea_ry_full(nqubits,nlayer):
    qubitlst = [i for i in range(nqubits)]
    xy = 'x'
    ansatz = hea_ry_pre(qubitlst,xy)
    for nth in range(1,nlayer+1):
        unit = hea_ry_full_unit(qubitlst,nth,xy)
        ansatz += unit
    return ansatz

# ry_linear
def hea_ry_linear_unit(qubitlst,nthlayer,xy):
    encoder = Circuit()
    for j in range(len(qubitlst)-1):
        encoder += X.on(qubitlst[j+1], qubitlst[j]) 
    for j in qubitlst:
        encoder += RY('l'+str(nthlayer)+xy+'1p'+str(j)).on(j)     
    return encoder

def hea_ry_linear(nqubits,nlayer):
    qubitlst = [i for i in range(nqubits)]
    xy = 'x'
    ansatz = hea_ry_pre(qubitlst,xy)
    for nth in range(1,nlayer+1):
        unit = hea_ry_linear_unit(qubitlst,nth,xy)
        ansatz += unit
    return ansatz

# ry_cascade
def hea_ry_cascade_unit(qubitlst,nthlayer,xy):
    encoder = Circuit()
    for j in range(len(qubitlst)-1):
        encoder += X.on(qubitlst[j+1], qubitlst[j])
    #encoder.barrier()    
    for j in qubitlst:
        encoder += RY('l'+str(nthlayer)+xy+'1p'+str(j)).on(j)
    #encoder.barrier() 
    for j in range(len(qubitlst)-1):
        encoder += X.on(qubitlst[::-1][j], qubitlst[::-1][j+1])
    #encoder.barrier()    
    for j in qubitlst:
        encoder += RY('l'+str(nthlayer)+xy+'2p'+str(j)).on(j)
    return encoder

def hea_ry_cascade(nqubits,nlayer):
    qubitlst = [i for i in range(nqubits)]
    xy = 'x'
    ansatz = hea_ry_pre(qubitlst,xy)
    for nth in range(1,nlayer+1):
        unit = hea_ry_cascade_unit(qubitlst,nth,xy)
        ansatz += unit
    return ansatz

# RyRz, EfficientSU2
def hea_RyRz_pre(qubitlst,xy):
    # Add Initial Gates
    pre = Circuit()
    for j in qubitlst:
        pre += RY('l0'+xy+'0p'+str(j)).on(j)
        pre += RZ('l0'+xy+'1p'+str(j)).on(j)
    return pre

def hea_RyRz_unit(qubitlst,nthlayer,xy):
    encoder = Circuit()
    for j in range(len(qubitlst)-1):
        for k in range(j+1,len(qubitlst)):
            encoder += X.on(qubitlst[k], qubitlst[j]) 
    for j in qubitlst:
        encoder += RY('l'+str(nthlayer)+'1p'+str(j)).on(j)
        encoder += RZ('l'+str(nthlayer)+'2p'+str(j)).on(j) 
    return encoder

def hea_RyRz(nqubits,nlayer):
    qubitlst = [i for i in range(nqubits)]
    xy = 'x'
    ansatz = hea_RyRz_pre(qubitlst,xy)
    for nth in range(1,nlayer+1):
        unit = hea_RyRz_unit(qubitlst,nth,xy)
        ansatz += unit
    return ansatz

# fSim_linear
def hea_fSim_linear_unit(qubitlst, nthlayer, xy):
    encoder = Circuit()
    for j in range(len(qubitlst)-1):
        theta = 'l'+str(nthlayer)+xy+'1p'+str(qubitlst[j])
        phi   = 'l'+str(nthlayer)+xy+'2p'+str(qubitlst[j])
        encoder += FSim(theta, phi).on([qubitlst[j], qubitlst[j+1]])
    return encoder

def hea_fSim_linear(nqubits,nlayer):
    qubitlst = [i for i in range(nqubits)]
    xy = 'x'
    ansatz = Circuit()
    for nth in range(1, nlayer+1):
        unit = hea_fSim_linear_unit(qubitlst, nth, xy)
        ansatz += unit
    return ansatz

# fSim_brickwall
def hea_fSim_brickwall_unit(qubitlst,nthlayer,xy):
    encoder = Circuit()
    for n in [0, 1]:
        for j in range(n, len(qubitlst)-1, 2):
            theta = 'l'+str(nthlayer)+xy+'1p'+str(qubitlst[j])
            phi   = 'l'+str(nthlayer)+xy+'2p'+str(qubitlst[j])
            encoder += FSim(theta, phi).on([qubitlst[j], qubitlst[j+1]])
    return encoder

def hea_fSim_brickwall(nqubits,nlayer):
    qubitlst = [i for i in range(nqubits)]
    xy = 'x'
    ansatz = Circuit()
    for nth in range(1, nlayer+1):
        unit = hea_fSim_brickwall_unit(qubitlst,nth,xy)
        ansatz += unit
    return ansatz

# ASWAP

def Agate(q1,q2,phi,theta):
    R = Circuit()
    R += RZ({phi:1}).on(q2)
    R += RZ(np.pi).on(q2)
    R += RY({theta:1}).on(q2)
    R += RY(np.pi/2).on(q2)

    Rd = R.hermitian()
    cir = Circuit()
    cir += X.on(q1,q2)
    cir += R
    cir += X.on(q2,q1)
    cir += Rd
    cir += X.on(q1,q2)
    return cir

def hea_ASWAP_unit(qubitlst,nthlayer,xy):
    cir = Circuit()
    for n in [0,1]:
        for i in range(n,len(qubitlst)-1,2):
            phi = 'l'+str(nthlayer)+xy+'1p'+str(qubitlst[i])
            theta = 'l'+str(nthlayer)+xy+'2p'+str(qubitlst[i])
            cir += Agate(qubitlst[i], qubitlst[i+1], phi, theta)
    return cir

def hea_ASWAP(nqubits,nlayer):
    qubitlst = [i for i in range(nqubits)]
    xy = 'x'
    ansatz = Circuit()
    for nth in range(1,nlayer+1):
        unit = hea_ASWAP_unit(qubitlst,nth,xy)
        ansatz += unit
    return ansatz

############################################
#
#                  PCHEA
#
############################################

# Vmat = np.array([[1+1j,1-1j],[1-1j,1+1j]])*0.5
# Vgate = UnivMathGate('V',Vmat)
# Vdgate = UnivMathGate('Vd',Vmat.conj().T)
# 
# def C2AB(q1,q2,pa,pb):
#     """decompose expontial gate: e^{-1j/2(alpha*XX+beta*YY)}"""
#     c2 = Circuit()
#     c2 += Z.on(q1)
#     c2 += Z.on(q2)
#     c2 += Vdgate.on(q1)
#     c2 += Vdgate.on(q2)
#     c2 += X.on(q2, q1)
#     c2 += RX({pa:1}).on(q1)
#     c2 += RZ({pb:1}).on(q2)
#     c2 += X.on(q2, q1)
#     c2 += Vgate.on(q1)
#     c2 += Vgate.on(q2)
#     c2 += Z.on(q1)
#     c2 += Z.on(q2)
#     return c2

def E2AB(q1,q2,alpha,beta):
    """e^{-1j/2(alpha*XX+beta*YY)}"""
    cir = Circuit()
    cir += Rxx({alpha:1}).on([q1,q2])
    cir += Ryy({beta:1}).on([q1,q2])
    return cir

def G2E(q1,q2,alpha,beta):
    g2 = Circuit()
    g2 += RY({alpha:1,beta:-1}).on(q1)
    g2.barrier()
    g2 += E2AB(q1,q2,alpha,beta)
    g2.barrier()
    g2 += RY({beta:1,alpha:-1}).on(q1)
    g2 += RX({beta:1,alpha:-1}).on(q2)
    g2 += RZ({beta:1,alpha:-1}).on(q1)
    return g2

def fSim1(q1,q2,theta):
    return E2AB(q1,q2,theta,theta)

def fSim2(q1,q2,phi):
    f = Circuit()
    f += X.on(q2,q1)
    f += RZ({phi:0.5}).on(q2)
    f += X.on(q2,q1)
    f += RZ({phi:-0.5}).on(q1)
    f += RZ({phi:-0.5}).on(q2)
    return f

def G2F_decomposed(q1,q2,phi,theta):
    """ Note the ordinary is (phi, theta)"""
    g2 = Circuit()
    g2 += RY({phi:-0.5}).on(q2)
    g2.barrier()
    g2 += fSim1(q1,q2,theta)
    g2.barrier()
    g2 += fSim2(q1,q2,phi)
    g2.barrier()
    g2 += RY({phi:0.5}).on(q2)
    return g2

def G2F(q1,q2,phi,theta):
    g2  = Circuit()
    g2 += RY({phi:-0.5}).on(q2)
    g2 += FSim(theta,phi).on([q1,q2])
    g2 += RY({phi:0.5}).on(q2)
    return g2


#          PCHEA


def part1(qubitlst,nthlayer,xy,gates='XY'):
    gates_dict = {'X':RX,'Y':RY,'Z':RZ}
    part1 = Circuit()
    if gates[0] in gates_dict:
        col1 = gates_dict[gates[0]]
        part1 += Circuit([col1('l'+str(nthlayer)+xy+'1p'+str(i)).on(i) for i in qubitlst])
    if gates[1] in gates_dict:
        col2 = gates_dict[gates[1]]
        part1 += Circuit([col2('l'+str(nthlayer)+xy+'2p'+str(i)).on(i) for i in qubitlst])
    #part1.barrier()
    return part1

def part2_F(qubitlst,nthlayer,xy):
    part2 = Circuit()
    for i in range(len(qubitlst)-1):
        phi   = 'l'+str(nthlayer)+xy+'3p'+str(qubitlst[i])
        theta = 'l'+str(nthlayer)+xy+'4p'+str(qubitlst[i])
        part2 += G2F(qubitlst[i],qubitlst[i+1],phi,theta)
    return part2

def part2_D(qubitlst,nthlayer,xy):
    part2 = Circuit()
    for i in range(len(qubitlst)-1):
        phi   = 'l'+str(nthlayer)+xy+'3p'+str(qubitlst[i])
        theta = 'l'+str(nthlayer)+xy+'4p'+str(qubitlst[i])
        part2 += G2F_decomposed(qubitlst[i],qubitlst[i+1],phi,theta)
    return part2

def part2_E(qubitlst,nthlayer,xy):
    part2 = Circuit()
    for i in range(len(qubitlst)-1):
        alpha = 'l'+str(nthlayer)+xy+'3p'+str(qubitlst[i])
        beta = 'l'+str(nthlayer)+xy+'4p'+str(qubitlst[i])
        part2 += G2E(qubitlst[i],qubitlst[i+1],alpha,beta)
    return part2

def part2_C(qubitlst):
    part2 = Circuit()
    for i in range(len(qubitlst)-1):
        part2 += X.on(qubitlst[i+1],qubitlst[i])
    return part2

def part2(qubitlst,nthlayer,xy,gates='F'):
    if gates == 'F':
        return part2_F(qubitlst,nthlayer,xy)
    elif gates == 'D':
        return part2_D(qubitlst,nthlayer,xy)
    elif gates == 'E':
        return part2_E(qubitlst,nthlayer,xy)
    elif gates == 'C':
        return part2_C(qubitlst)

def part3(qubitlst,nthlayer,xy,gates='Y2'):
    gates_dict = {'X':RX,'Y':RY,'Z':RZ}
    col3 = gates_dict[gates[0]]
    if gates[1] == '1':
        part3 = Circuit([col3('l'+str(nthlayer)+xy+'5p'+str(qubitlst[-1])).on(qubitlst[-1])])
    elif gates[1] == '2':
        part3 = Circuit([col3('l'+str(nthlayer)+xy+'5p'+str(i)).on(i) for i in qubitlst])
    return part3

def pchea_unit(qubitlst,nthlayer,xy,gate_orden):
    cir1 = part1(qubitlst,nthlayer,xy,gate_orden[:2])
    cir2 = part2(qubitlst,nthlayer,xy,gate_orden[-1])
    cir3 = part3(qubitlst,nthlayer,xy,gate_orden[2:4])
    
    unit_left = cir1+cir2 
    unit_right = unit_left.hermitian()
    unit = unit_left + cir3 + unit_right
    return unit

def pchea(nqubits,nlayer,gate_orden):
    """ gate_orden: 'XYZ1F','XYZ2F','YXZ1F','YXZ2F','YXY1F','YXY2F' 
                    'XYZ1E','XYZ2E','YXZ1E','YXZ2E','YXY1E','YXY2E',……
        Two-qubit-gates: F, D (decomposed fSim), E , C (CNOT)
    """
    ansatz = Circuit()
    qubitlst = [i for i in range(nqubits)]
    xy = 'x'
    for nth in range(1,nlayer+1):
        unit = pchea_unit(qubitlst,nth,xy,gate_orden)
        ansatz += unit
    return ansatz