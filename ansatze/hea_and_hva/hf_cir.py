import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import I,H,X,CNOT
from mindquantum.core.operators import QubitOperator

############################################
#
#              HF reference 
#
############################################

def hf_input(nqubits,nelec,ref):
    
    """
    take 'HF' as reference in general,
    or take 'SITE' or 'OAO' basis as reference for Hubbard Model,
    or take 'LOCAL' basis as reference for local (H2)n,
    or None ,
    or define by yourself with a list of occupied index,
    """
    
    def ref_simple(nqubits,HF_initial):
        hf = Circuit([I.on(i) for i in range(nqubits)])
        hf += Circuit([X.on(i) for i in HF_initial])
        hf.barrier()
        return hf
    
    if type(ref) == str:
        Nalpha, Nbeta = nelec 
        if ref == 'HF':
            HF_initial = np.arange(0,2*Nalpha,2).tolist()+np.arange(1,2*Nbeta+1,2).tolist()
            return ref_simple(nqubits,HF_initial) 
        elif ref == 'SITE' or ref == 'OAO':
            HF_initial = np.arange(0,4*Nalpha,4).tolist()+np.arange(3,4*Nbeta,4).tolist()
            return ref_simple(nqubits,HF_initial) 
        elif ref == 'LOCAL':
            h2_na,h2_nb = (1,1)
            h2_qubits = 4
            HF_initial = []
            for idx in range(Nalpha):
                h2n = (np.arange(0,2*h2_na,2)+h2_qubits*idx).tolist()\
                +(np.arange(1,2*h2_nb+1,2)+h2_qubits*idx).tolist()
                HF_initial += h2n
            return ref_simple(nqubits,HF_initial) 
        elif ref == 'None' or ref=='':
            HF_initial = []
            return ref_simple(nqubits,HF_initial)
        elif ref=='Neel':
            HF_initial = range(0,nqubits,2)
            return ref_simple(nqubits,HF_initial)
        elif ref=='Bell':
            cir = Circuit([X.on(i) for i in range(nqubits)])
            cir += Circuit([H.on(2*i) for i in range(nqubits//2)])
            cir += Circuit([CNOT.on(2*i+1,2*i) for i in range(nqubits//2)]) 
            return cir
        elif ref=='allH':
            cir = Circuit([H.on(i) for i in range(nqubits)])
            return cir
        else:
            raise TypeError('bad type')
        
    elif type(ref) == list or type(ref) ==range:
        HF_initial = ref
        return ref_simple(nqubits,HF_initial)
    
    elif type(ref) == tuple:
        # for XYZ2F, aniso ham only
        hea_type = 'XYZ2F'
        HF_initial,amps = ref
        nparams0 = get_hea_nparams(nqubits, 1, hea_type)
        nlayer = len(amps)//nparams0
        print('load ',nlayer,'XYZ2F as reference')
        cir = ref_simple(nqubits,HF_initial)
        for nthlayer in range(1,nlayer+1):
            cir += get_hea_unit(nqubits, nthlayer, hea_type)
        pr = dict(zip(cir.params_name,amps))
        return cir.apply_value(pr=pr)
    else:
        raise TypeError('bad type')
        print('Attention!')
        print('Support fromat: str, list, tuple(list,amps) only for XYZ2F')
        return None