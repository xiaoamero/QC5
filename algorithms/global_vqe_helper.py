import numpy as np
from mindquantum.core.operators import QubitOperator

# 
#       intiallize for phi_hea from unitary ccsd ansatz qubit strings
#

def tcuck_hea_amps_from_small_to_big(nqubits0,nlayer0,amps_array0,nqubits1,initial_type='0',
                                      nlayer1=None):
    # 0: small 1:big
    hea_type = 'XYZ2F'
    if nlayer1 == None:
        nlayer1 = nlayer0
    print(nlayer0,'layers amps were encoding')
    nparams0 = get_hea_nparams(nqubits0,nlayer0,hea_type)
    ndarray0 = amps_array0[:nparams0].reshape((nlayer0,-1))
    
    # econding
    nparams2_ = get_hea_nparams(nqubits1,1,hea_type)
    ndarray2 = np.zeros((nlayer1,nparams2_))
    for idx in range(0,nlayer0):
        col01 = ndarray0[idx,:nqubits0]
        col02 = ndarray0[idx,nqubits0:2*nqubits0]
        col03 = ndarray0[idx,2*nqubits0:4*nqubits0-2]
        col04 = ndarray0[idx,4*nqubits0-2:]  
        
        ndarray2[idx,:nqubits0] = col01
        ndarray2[idx,nqubits1:nqubits1+nqubits0] = col02
        ndarray2[idx,2*nqubits1:2*nqubits1+2*nqubits0-2] = col03
        ndarray2[idx,4*nqubits1-2:4*nqubits1-2+nqubits0] = col04
    if initial_type == '0':  
        return ndarray2.reshape(-1)
    elif initial_type == 'H':
        ndarray2[0,nqubits0] = np.pi/2
        ndarray2[0,4*nqubits1-2+nqubits0] = np.pi/2
        return ndarray2.reshape(-1)
    elif initial_type == 'HH':
        ndarray2[0,nqubits0] = np.pi/2
        ndarray2[0,nqubits1-1] = np.pi/2
        ndarray2[0,4*nqubits1-2+nqubits0] = np.pi/2
        ndarray2[0,4*nqubits1-2+nqubits1-1] = np.pi/2
        return ndarray2.reshape(-1)
    
def seperate_phi_hea_layers(nqubits,nlayer,amps_array,hea_type):
    nparams0 = get_hea_nparams(nqubits,nlayer,hea_type)
    amps_ndarray = amps_array[:nparams0].reshape((nlayer,-1))
    # econding
    for idx in range(nlayer):
        print('seperate PHIHEA amplitudes, layer=',idx+1)
        print('col1',amps_ndarray[idx,:nqubits])
        print('col2',amps_ndarray[idx,nqubits:2*nqubits])
        print('col3',amps_ndarray[idx,2*nqubits:4*nqubits-2])
        print('col4',amps_ndarray[idx,4*nqubits-2:])
    return None

def combine_hea_layers(nqubits0,amps_array0,nqubits1,amps_array1,nlayer,hea_type):
    if hea_type in ['XYZ2F']:
        # 0: first 1:last
        nparams0 = get_hea_nparams(nqubits0,nlayer,hea_type)
        nparams1 = get_hea_nparams(nqubits1,nlayer,hea_type)
        
        ndarray0 = amps_array0[:nparams0].reshape((nlayer,-1))
        ndarray1 = amps_array1[:nparams1].reshape((nlayer,-1))
        # econding
        nqubits2 = nqubits0+nqubits1
        nparams2_ = get_hea_nparams(nqubits2,1,hea_type)
        ndarray2 = np.zeros((nlayer,nparams2_))
        for idx in range(nlayer):
            col01 = ndarray0[idx,:nqubits0]
            col02 = ndarray0[idx,nqubits0:2*nqubits0]
            col03 = ndarray0[idx,2*nqubits0:4*nqubits0-2]
            col04 = ndarray0[idx,4*nqubits0-2:]
            
            col11 = ndarray1[idx,:nqubits1]
            col12 = ndarray1[idx,nqubits1:2*nqubits1]
            col13 = ndarray1[idx,2*nqubits1:4*nqubits1-2]
            col14 = ndarray1[idx,4*nqubits1-2:]        
            
            ndarray2[idx,:nqubits2] = np.concatenate((col01,col11))
            ndarray2[idx,nqubits2:2*nqubits2] = np.concatenate((col02,col12))
            col00 = np.zeros(2) # set G2 between two system as Identity
            ndarray2[idx,2*nqubits2:4*nqubits2-2] = np.concatenate((col03,col00,col13))
            ndarray2[idx,4*nqubits2-2:] = np.concatenate((col04,col14))
        return ndarray2.reshape(-1)
    elif hea_type in ['ry_linear','ry_full','EfficientSU2','ry_cascade']:
        amps_ndarray0 = amps_array0.reshape((nqubits0,-1))
        amps_ndarray1 = amps_array1.reshape((nqubits1,-1))
        amps_ndarray = np.concatenate((amps_array0,amps_array1))
        return amps_ndarray.reshape(-1)
    elif hea_type in ['ASWAP']:
        amps_ndarray0 = amps_array0.reshape((nlayer,-1))
        amps_ndarray0L = amps_ndarray0[:,:2*(nqubits0//2)]
        amps_ndarray0R = amps_ndarray0[:,2*(nqubits0//2):]
        
        amps_ndarray1 = amps_array1.reshape((nlayer,-1))
        amps_ndarray1L = amps_ndarray1[:,:2*(nqubits1//2)]
        if nqubits1 > 2:
            amps_ndarray1R = amps_ndarray1[:,2*(nqubits1//2):] 
        
        nqubits2 = nqubits0+nqubits1
        nparams = get_hea_nparams(nqubits2,1,hea_type)
        amps_ndarray2 = np.zeros((nlayer,nparams))
        amps_ndarray2L = amps_ndarray2[:,:2*(nqubits2//2)]
        amps_ndarray2R = amps_ndarray2[:,2*(nqubits2//2):]         
        for nth in range(nlayer):
            amps_ndarray2L[nth,:2*(nqubits0//2)] = amps_ndarray0L[nth]
            amps_ndarray2L[nth,-2*(nqubits1//2):]= amps_ndarray1L[nth]
            
            amps_ndarray2R[nth,:2*((nqubits0-1)//2)] = amps_ndarray0R[nth]
            if nqubits1>2:
                amps_ndarray2R[nth,-2*((nqubits1-1)//2):]= amps_ndarray1R[nth]
        
        amps_ndarray2 = np.concatenate((amps_ndarray2L,amps_ndarray2R),axis=1)
        return amps_ndarray2.reshape(-1) 

#
# non-interaction systems params copy, site4->site8
#
def site4Tosite8_ndarray(nqubits1,hea_type,collect_amps):
    nlayer = collect_amps.shape[0]
    nparams = get_hea_nparams(2*nqubits1,nlayer,hea_type)
    initial_ndarray = np.zeros((nlayer,nparams))
    for nth in range(nlayer):
        params_array1 = collect_amps[nth]
        params_array2 = copy_params_twice(nqubits1,nth+1,hea_type,params_array1)
        initial_ndarray[nth][:len(params_array2)] = params_array2
    return initial_ndarray    

def copy_params_twice(nqubits1,nlayer,hea_type,params_array1):
    """ nqubits1: the initial qubits number """
    num1 = get_hea_nparams(nqubits1,nlayer,hea_type)
    params_array1 = params_array1[:num1]
    
    if hea_type in ['ry_linear','ry_full','ry_cascade','EfficientSU2']:
        params_ndarray1 = params_array1.reshape((-1,nqubits1))
        params_ndarray2 = np.concatenate((params_ndarray1,params_ndarray1),axis=1)
        return params_ndarray2.reshape(-1)
    elif hea_type in ['ASWAP']:
        params_ndarray1 = params_array1.reshape((nlayer,-1))
        num2 = get_hea_nparams(2*nqubits1, 1, hea_type)
        params_ndarray2 = np.zeros((nlayer,num2))
        for nth in range(nlayer):
            params_ndarray2[nth,:nqubits1] = params_ndarray1[nth,:nqubits1]
            params_ndarray2[nth,nqubits1:2*nqubits1] = params_ndarray1[nth,:nqubits1]
            params_ndarray2[nth,2*nqubits1:3*nqubits1-2] = params_ndarray1[nth,nqubits1:]
            params_ndarray2[nth,3*nqubits1:] = params_ndarray1[nth,nqubits1:]
        return params_ndarray2.reshape(-1)
    elif hea_type in ['YXY2F','XYZ2F']:
        params_ndarray1 = params_array1.reshape((nlayer,-1))
        nqubits2 = nqubits1*2
        num2 = get_hea_nparams(nqubits2, 1, hea_type)
        params_ndarray2 = np.zeros((nlayer,num2))
        for nth in range(nlayer):
            unit1 = params_ndarray1[nth,:nqubits1]
            params_ndarray2[nth,:nqubits2] = np.concatenate((unit1,unit1))
            unit2 = params_ndarray1[nth,nqubits1:2*nqubits1]
            params_ndarray2[nth,nqubits2:2*nqubits2] = np.concatenate((unit2,unit2))
            unit3 = params_ndarray1[nth,2*nqubits1:4*nqubits1-2]
            params_ndarray2[nth,2*nqubits2:4*nqubits2-2] = np.concatenate((unit3,np.zeros(2),unit3))
            unit4 = params_ndarray1[nth,4*nqubits1-2:5*nqubits1-2]
            params_ndarray2[nth,(4*nqubits2-2):] = np.concatenate((unit4,unit4))
            
            # XXXIAO make the G2F between the two site4u1 = CNOT,
            # params_ndarray2[nth,(3*nqubits2-2):] = np.pi #　
        return params_ndarray2.reshape(-1)
#
# cascade to YXY2F
#
def CascadeToYXY2F(cascade_amps,cascade_nlayer,nqubits):
    cascade_ndarray = cascade_amps.reshape((-1,nqubits))
    
    para_1 = get_hea_nparams(nqubits,1,'YXY2F')
    yxy2f_nlayer = cascade_nlayer+1
    yxy2f = np.zeros((yxy2f_nlayer,para_1))
    # encoding CNOT nlayer:1-(yxy2f_nlayer-1)
    cnot_param = np.zeros(nqubits*2-2)
    cnot_param[::2]=np.pi
    for i in range(yxy2f_nlayer-1):
        yxy2f[i][2*nqubits:(4*nqubits-2)] = cnot_param 
    for i in range(yxy2f_nlayer-1):
        # ry1
        for j in range(i+1):
            yxy2f[i][:nqubits] += cascade_ndarray[2*j]    
        # ry2
        yxy2f[i][-nqubits:] = cascade_ndarray[2*i+1]
    # last layer
    for i in range(yxy2f_nlayer):
        yxy2f[-1][-nqubits:] += cascade_ndarray[2*i]
    return yxy2f.reshape(-1)

def gen_cascade_to_YXY2F_ndarray(nqubits,cascade_nlayer,cascade_amps):
    assert(cascade_nlayer<=cascade_amps.shape[0]) 
    nparams = get_hea_nparams(nqubits, cascade_nlayer+1,'YXY2F')
    initial_ndarray = np.zeros((cascade_nlayer,nparams))
    for cascade_nthlayer in range(1,cascade_nlayer+1): 
        para_2 = get_hea_nparams(nqubits,cascade_nthlayer,'ry_cascade')
        initial = CascadeToYXY2F(cascade_amps[cascade_nthlayer-1][:para_2],cascade_nthlayer,nqubits)
        initial_ndarray[cascade_nthlayer-1][:len(initial)] = initial
    return initial_ndarray

def CascadeToYIY2F(cascade_amps,cascade_nlayer,nqubits):
    cascade_ndarray = cascade_amps.reshape((-1,nqubits))
    
    para_1 = get_hea_nparams(nqubits,1,'YIY2F')
    yxy2f_nlayer = cascade_nlayer+1
    yxy2f = np.zeros((yxy2f_nlayer,para_1))
    # encoding CNOT nlayer:1-(yxy2f_nlayer-1)
    cnot_param = np.zeros(nqubits*2-2)
    cnot_param[::2]=np.pi
    for i in range(yxy2f_nlayer-1):
        yxy2f[i][nqubits:(3*nqubits-2)] = cnot_param 
    for i in range(yxy2f_nlayer-1):
        # ry1
        for j in range(i+1):
            yxy2f[i][:nqubits] += cascade_ndarray[2*j]    
        # ry2
        yxy2f[i][-nqubits:] = cascade_ndarray[2*i+1]
    # last layer
    for i in range(yxy2f_nlayer):
        yxy2f[-1][-nqubits:] += cascade_ndarray[2*i]
    return yxy2f.reshape(-1)

def gen_cascade_to_YIY2F_ndarray(nqubits,cascade_nlayer,cascade_amps):
    assert(cascade_nlayer<=cascade_amps.shape[0]) 
    nparams = get_hea_nparams(nqubits, cascade_nlayer+1,'YIY2F')
    initial_ndarray = np.zeros((cascade_nlayer,nparams))
    for cascade_nthlayer in range(1,cascade_nlayer+1): 
        para_2 = get_hea_nparams(nqubits,cascade_nthlayer,'ry_cascade')
        initial = CascadeToYIY2F(cascade_amps[cascade_nthlayer-1][:para_2],cascade_nthlayer,nqubits)
        initial_ndarray[cascade_nthlayer-1][:len(initial)] = initial
    return initial_ndarray

def YIY2C_to_YIY2F(nqubits,nlayer,YIY2C_amps):
    f_nparams = get_hea_nparams(nqubits,1,'YIY2F')
    f_amps = np.zeros((nlayer,f_nparams))
    c_amps = YIY2C_amps.reshape((nlayer,-1))
    
    for nth in range(nlayer):
        # CNOT 
        for i in range(nqubits,3*nqubits-2,2):
            f_amps[nth][i] = np.pi
        # ry1
        f_amps[nth][:nqubits] = c_amps[nth][:nqubits]
        # ry2
        f_amps[nth][-nqubits:] = c_amps[nth][-nqubits:]
    return f_amps.reshape(-1)

def gen_YIY2C_to_YIY2F_ndarray(nqubits,nlayer,YIY2C_amps):
    nparams = get_hea_nparams(nqubits, nlayer,'YIY2F')
    initial_ndarray = np.zeros((nlayer,nparams))
    for nthlayer in range(1,nlayer+1): 
        para_2 = get_hea_nparams(nqubits,nthlayer,'YIY2C')
        initial = YIY2C_to_YIY2F(nqubits,nthlayer,YIY2C_amps[nthlayer-1][:para_2])
        initial_ndarray[nthlayer-1][:len(initial)] = initial
    return initial_ndarray    

def YIY2F_to_YXY2F(nqubits,nlayer,YIY2F_amps):
    f_nparams = get_hea_nparams(nqubits,1,'YXY2F')
    f_amps = np.zeros((nlayer,f_nparams))
    c_amps = YIY2F_amps.reshape((nlayer,-1))
    
    for nth in range(nlayer):
        # ry1
        f_amps[nth][:nqubits] = c_amps[nth][:nqubits]
        # ry2
        f_amps[nth][2*nqubits:] = c_amps[nth][nqubits:]
    return f_amps.reshape(-1)

def gen_YIY2F_to_YXY2F_ndarray(nqubits,nlayer,YIY2C_amps):
    nparams = get_hea_nparams(nqubits, nlayer,'YXY2F')
    initial_ndarray = np.zeros((nlayer,nparams))
    for nthlayer in range(1,nlayer+1): 
        para_2 = get_hea_nparams(nqubits,nthlayer,'YIY2F')
        initial = YIY2F_to_YXY2F(nqubits,nthlayer,YIY2C_amps[nthlayer-1][:para_2])
        initial_ndarray[nthlayer-1][:len(initial)] = initial
    return initial_ndarray

def YIY2C_to_YXY2C(nqubits,nlayer,YIY2F_amps):
    f_nparams = get_hea_nparams(nqubits,1,'YXY2C')
    f_amps = np.zeros((nlayer,f_nparams))
    c_amps = YIY2F_amps.reshape((nlayer,-1))
    
    for nth in range(nlayer):
        # ry1
        f_amps[nth][:nqubits] = c_amps[nth][:nqubits]
        # ry2
        f_amps[nth][2*nqubits:] = c_amps[nth][nqubits:]
    return f_amps.reshape(-1)

def gen_YIY2C_to_YXY2C_ndarray(nqubits,nlayer,YIY2C_amps):
    nparams = get_hea_nparams(nqubits, nlayer,'YXY2C')
    initial_ndarray = np.zeros((nlayer,nparams))
    for nthlayer in range(1,nlayer+1): 
        para_2 = get_hea_nparams(nqubits,nthlayer,'YIY2C')
        initial = YIY2C_to_YXY2C(nqubits,nthlayer,YIY2C_amps[nthlayer-1][:para_2])
        initial_ndarray[nthlayer-1][:len(initial)] = initial
    return initial_ndarray 

def YXY2C_to_YXY2F(nqubits,nlayer,YXY2C_amps):
    f_nparams = get_hea_nparams(nqubits,1,'YXY2F')
    f_amps = np.zeros((nlayer,f_nparams))
    c_amps = YXY2C_amps.reshape((nlayer,-1))
    
    for nth in range(nlayer):
        # CNOT 
        for i in range(2*nqubits,4*nqubits-2,2):
            f_amps[nth][i] = np.pi
        # ry1
        f_amps[nth][:2*nqubits] = c_amps[nth][:2*nqubits]
        # ry2
        f_amps[nth][-nqubits:] = c_amps[nth][-nqubits:]
    return f_amps.reshape(-1)

def gen_YXY2C_to_YXY2F_ndarray(nqubits,nlayer,YXY2C_amps):
    nparams = get_hea_nparams(nqubits, nlayer,'YXY2F')
    initial_ndarray = np.zeros((nlayer,nparams))
    for nthlayer in range(1,nlayer+1): 
        para_2 = get_hea_nparams(nqubits,nthlayer,'YXY2C')
        initial = YXY2C_to_YXY2F(nqubits,nthlayer,YXY2C_amps[nthlayer-1][:para_2])
        initial_ndarray[nthlayer-1][:len(initial)] = initial
    return initial_ndarray    

#
#   ADAPT Helper
#

#
# qubitOp to XYZ2F/XYZ1F
#

def qubitAdaptVqe2PhiHeaInitial(nqubits,adapt_opA_pool,adapt_amps,phi_hea_type,encoding='every_layer'):
    """ adapt_opA_pool: just read adapt result file
        adapt_amps: just read adapt result file 
    """
    Ftype = ['phi_heaF1','phi_heaF2','XYZ1F','XYZ2F']
    Etype = ['phi_heaE1','phi_heaE2','XYZ1E','XYZ2E']
    if phi_hea_type in Ftype:
        amps_collect = []
        for idx,qop in enumerate(adapt_opA_pool):
            amps_array = SinglePauliAmplitudeToParameterFtype(nqubits,phi_hea_type,qop,idx+1)
            amps_collect.append(amps_array)
    elif phi_hea_type in Etype:
        amps_collect = []
        for idx,qop in enumerate(adapt_opA_pool):
            amps_array = SinglePauliAmplitudeToParameterEtype(nqubits,phi_hea_type,qop,idx+1)
            amps_collect.append(amps_array)
    initial_ndarray = np.array(amps_collect)
    if encoding == 'every_layer':
        nlayer = len(adapt_opA_pool)
        nparams = get_hea_nparams(nqubits,nlayer,phi_hea_type)
        new_amps = np.zeros((nlayer,nparams))
        for idx in range(nlayer):
            xox = initial_ndarray[:idx+1]
            xox[:,-1] = 2*adapt_amps[idx]
            xox = xox.reshape(-1)
            new_amps[idx][:len(xox)] = xox
        return new_amps
    elif encoding == 'last_layer':
        initial_ndarray[:,-1] = 2*adapt_amps[-1]
        return initial_ndarray
    
def SinglePauliAmplitudeToParameterFtype(nqubits,phi_hea_type,qubitOp,nthlayer=1,printdetails=True):
    pi = np.pi
    if phi_hea_type == 'phi_heaF1' or phi_hea_type=='XYZ1F':
        packed_array = np.zeros(nqubits*4-1)
    elif phi_hea_type == 'phi_heaF2' or phi_hea_type=='XYZ2F':
        packed_array = np.zeros(nqubits*5-2)
    for k,v in qubitOp.terms.items(): # if need sort version just loop param_sorted
        if printdetails:
            print('layer',nthlayer,round(abs(v),3),k)
        # start enconding ……
        # RX RY
        pauliIndex = []
        for i in k:
            if i[1] == 'X':
                packed_array[i[0]] = 0
                packed_array[nqubits+i[0]] = -pi/2
            elif i[1] == 'Y':
                packed_array[i[0]] = pi/2
                packed_array[nqubits+i[0]] = 0
            elif i[1] == 'Z':
                packed_array[i[0]] = 0
                packed_array[nqubits+i[0]] = 0
            pauliIndex.append(i[0])  
        #print(pauliIndex)
        # Fsim
        mi = min(pauliIndex)
        ma = max(pauliIndex)
        for i in range(mi):
            #print('iden',2*nqubits+2*i,2*nqubits+2*i+1)
            packed_array[2*nqubits+2*i] = 0
            packed_array[2*nqubits+2*i+1] = 0 
        for i in range(ma,nqubits-1):
            #print('iswap',2*nqubits+2*i,2*nqubits+2*i+1)
            packed_array[2*nqubits+2*i] = 0   # theta
            packed_array[2*nqubits+2*i+1] = -pi/2 # phi
        for idx,i in enumerate(pauliIndex):
            if idx+1<len(pauliIndex):
                while i+1 < pauliIndex[idx+1]:
                    #print('iswap',2*nqubits+2*i,2*nqubits+2*i+1)
                    packed_array[2*nqubits+2*i] = 0   # theta
                    packed_array[2*nqubits+2*i+1] = -pi/2 # phi 
                    i += 1
                #print('cnot',2*nqubits+2*i,2*nqubits+2*i+1)
                packed_array[2*nqubits+2*i] = pi   # theta
                packed_array[2*nqubits+2*i+1] = 0 # phi
        # shit code but running well        
        # RZ
        if type(v) == complex:
            v = v.imag
        packed_array[-1] = v*2 # exp(-iHt)
    return packed_array

def SinglePauliAmplitudeToParameterEtype(nqubits,phi_hea_type,qubitOp,nthlayer=1,printdetails=True):
    pi = np.pi
    if phi_hea_type == 'phi_heaE1' or phi_hea_type == 'XYZE1':
        packed_array = np.zeros(nqubits*4-1)
    elif phi_hea_type == 'phi_heaE2' or phi_hea_type == 'XYZE2':
        packed_array = np.zeros(nqubits*5-2)
    for k,v in qubitOp.terms.items(): # if need sort version just loop param_sorted
        if printdetails:
            print('layer',nthlayer,round(abs(v),3),k)
        # start enconding ……
        # RX RY
        pauliIndex = []
        for i in k:
            if i[1] == 'X':
                packed_array[i[0]] = 0
                packed_array[nqubits+i[0]] = -pi/2
            elif i[1] == 'Y':
                packed_array[i[0]] = pi/2
                packed_array[nqubits+i[0]] = 0
            elif i[1] == 'Z':
                packed_array[i[0]] = 0
                packed_array[nqubits+i[0]] = 0
            pauliIndex.append(i[0])  
        #print(pauliIndex)
        # G2
        mi = min(pauliIndex)
        ma = max(pauliIndex)
        for i in range(mi):
            #print('iden',2*nqubits+2*i,2*nqubits+2*i+1)
            packed_array[2*nqubits+2*i] = 0
            packed_array[2*nqubits+2*i+1] = 0 
        for i in range(ma,nqubits-1):
            #print('iswap',2*nqubits+2*i,2*nqubits+2*i+1)
            packed_array[2*nqubits+2*i] = pi/2   # theta
            packed_array[2*nqubits+2*i+1] = pi/2 # phi
        for idx,i in enumerate(pauliIndex):
            if idx+1<len(pauliIndex):
                while i+1 < pauliIndex[idx+1]:
                    #print('iswap',2*nqubits+2*i,2*nqubits+2*i+1)
                    packed_array[2*nqubits+2*i] = pi/2   # theta
                    packed_array[2*nqubits+2*i+1] = pi/2 # phi 
                    i += 1
                #print('cnot',2*nqubits+2*i,2*nqubits+2*i+1)
                packed_array[2*nqubits+2*i] = pi/2   # theta
                packed_array[2*nqubits+2*i+1] = 0 # phi
        # shit code but running well        
        # RZ
        if type(v) == complex:
            v = v.imag
        packed_array[-1] = v*2 # exp(-iHt)
    return packed_array

def PauliStringAmplitudeToQubitPool(qubitOp,sort=False):
    if sort:
        qubit_pool = []
        param_sorted = sorted(qubitOp.terms.items(),\
                              key = lambda x: abs(x[1]),reverse=True)  
        for k,v in param_sorted:
            qop = QubitOperator((k))*v
            qubit_pool.append(qop)
        return qubit_pool
    else:
        qubit_pool = []
        for k,v in qubitOp.terms.items():
            qop = QubitOperator((k))*v
            qubit_pool.append(qop)
        return qubit_pool

#
#     analyze
#

def analyze_hea_psi(nelec, circuit, input_amps=None, coeff_thres=0.1):
    """analyze the wavefunction into occupation number vectors"""
    # get psi from input_amps
    pr = dict(zip(circuit.params_name,input_amps))
    psi = circuit.get_qs(pr=pr, ket=False)
    # analyze psi
    n_qubits = circuit.n_qubits
    Nalpha, Nbeta = nelec
    # check 
    check=0
    for i in range(2**n_qubits):          
        tmp = "{:0"+str(n_qubits)+"b}"
        xt  = list(tmp.format(i))
        for idx,j in enumerate(xt):
            xt[idx] = eval(j)
        xxt = numpy.array(xt)
        even_sum = numpy.sum(xxt[::2]) 
        odd_sum = numpy.sum(xxt[1::2])
        if even_sum == Nalpha and odd_sum == Nbeta:
            txt = psi[i]
            check += numpy.abs(txt)**2
            if abs(txt)>=coeff_thres:
                print("ONVs:{} coeff:{: .9f} population:{:.4%}"\
                      .format(str(list(xt)),txt,numpy.abs(txt)**2))
    ok = numpy.isclose(check,1.0)
    if ok == False:
        print('Notice: the sum of population is ',check,'Please check your code (T=T)')
    return None