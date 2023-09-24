import numpy as np

def test_cal_ndepth(nqubits,nlayer,hea_type,size=(0,0)):
    from qiskit import QuantumCircuit
    from mindquantum.io.qasm import OpenQASM
    openqasm = OpenQASM()
    if hea_type in ['ry_linear','ry_full','ry_cascade','EfficientSU2']:
        hea = get_hea_ansatz(nqubits, size, nlayer, hea_type, ref='')
        pr = dict(zip(hea.params_name,np.ones(len(hea.params_name))))
        hea = hea.apply_value(pr = pr)
        
        ansatz_str = openqasm.to_string(hea)
        new_qc = QuantumCircuit.from_qasm_str(ansatz_str)
        depth = new_qc.depth()
        return depth
    
def check_params_orden(cir):
    params_name = cir.params_name
    new_order = []
    for temp in params_name:
        temp = temp.split('p',) # note the params_name is 'p'
        ordin = eval(temp[1])
        new_order.append(ordin)
    ok = np.allclose(np.array(new_order),np.arange(len(params_name)))
    print(ok)
    return None

def get_hva_heisenberg_pbc_ncnot(size,nlayer,pbc=True):
    # 3*nx*(ny-1)+3*ny*(nx-1) + 3*(nx+ny) in pbc general 
    nx,ny = size
    idx=0
    for ix in range(nx):
        for iy in range(ny-1):
            idx+=1
        if pbc and ny>2:
            idx+=1
    for iy in range(ny):
        for ix in range(nx-1):
            idx+=1
        if pbc and nx>2:
            idx+=1
    return idx*3*nlayer

def brickwall_depth(size,nlayer):
    nx,ny = size
    # 1D
    if nx==1 and ny==2:
        return 1*nlayer
    elif nx==1 and ny>2:
        return 2*nlayer
    elif ny==1 and nx ==2:
        return 1*nlayer
    elif ny==1 and nx >2:
        return 2*nlayer
    # 2D
    elif nx==2 and ny==2:
        return 2*nlayer
    elif nx==2 and ny>2:
        return 3*nlayer
    elif ny==2 and nx>2:
        return 3*nlayer
    elif nx>2 and ny>2:
        return 4*nlayer
    
def hva_heisenberg_ndepth(size,nlayer):
    nx,ny = size
    # 1D
    if nx==1 and ny==2:
        return 3*nlayer
    elif nx==1 and ny>2:
        return 6*nlayer
    elif ny==1 and nx ==2:
        return 3*nlayer
    elif ny==1 and nx >2:
        return 6*nlayer
    # 2D
    elif nx==2 and ny==2:
        return 3*nlayer*2
    elif nx==2 and ny>2:
        return 3*nlayer+6*nlayer
    elif ny==2 and nx>2:
        return 3*nlayer+6*nlayer
    elif nx>2 and ny>2:
        return 6*nlayer*2
    
def ry_linear_ndepth(nqubits,nlayer):
    if nqubits>=3:
        diff = (nqubits-2-1) # qubits 数不一样交错的程度也不一样
        return nqubits+1+(nqubits-diff)*(nlayer-1)
    elif nqubits==2:
        return 2*nlayer+1
    
def ry_full_ndepth(nqubits,nlayer):
    if nqubits>=3:
        diff = (nqubits-2-1) # qubits 数不一样交错的程度也不一样
        return (2*nqubits-2)+1+(2*nqubits-2-diff)*(nlayer-1)
    elif nqubits==2:
        return 2*nlayer+1  
    
def ry_cascade_ndepth(nqubits,nlayer):
    return 2*nqubits*nlayer+1

def EfficientSU2_ndepth(nqubits,nlayer):
    if nqubits>=3:
        diff = (nqubits-2-1) # qubits 数不一样交错的程度也不一样
        return (2*nqubits-1)+2+(2*nqubits-1-diff)*(nlayer-1)
    elif nqubits==2:
        return 3*nlayer+2
    
def get_pchea_nparams(nqubits,nlayer,gate_orden):
    gates1 = ['X','Y','Z']
    gates2 = ['D','E','F','C']
    assert(gate_orden[0] in gates1)
    assert(gate_orden[1] in gates1)
    assert(gate_orden[2] in gates1)
    assert(gate_orden[4] in gates2)
    if gate_orden[3] == '2':
        return (5*nqubits-2)*nlayer
    elif gate_orden[3] == '1':
        return (4*nqubits-1)*nlayer    