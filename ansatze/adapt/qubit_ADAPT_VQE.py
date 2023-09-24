import numpy as np
from mindquantum.core.circuit import decompose_single_term_time_evolution
from mindquantum.core.operators import QubitOperator, FermionOperator, TimeEvolution
from mindquantum.algorithm.nisq.chem.transform import Transform
from mindquantum.core.operators.utils import hermitian_conjugated
from QC_master.q1_ansatz import *
from QC_master.q2_xyz.PQC_Funs import get_expect_value_v2,get_scipy_optimized_vqe
from QC_master.q2_xyz.FerOp_QubitOp import get_qubit_opVdV
from functools import partial
from multiprocessing import Pool

#
# fermionic pool helper
#
def get_ref_idx(nqubits):
    refA = str()
    refB = str()
    for i in range(nqubits):
        if i%2 == 0:
            refA += '0'
            refB += '1'
        else:
            refA += '1'
            refB += '0'
    return int(refA,2),int(refB,2)

def check_maped_op_community(op):
    if type(op)==FermionOperator:
        qop = Transform(op-hermitian_conjugated(op)).jordan_wigner()
    elif type(op)==QubitOperator:
        qop = op
    
    ql = []
    for k,v in qop.terms.items():
        qq = QubitOperator(k)*1j
        ql.append(qq)
        
    cc = []
    num = len(ql)
    for i in range(num):
        for j in range(i,num):
            com = ql[i]*ql[j] - ql[j]*ql[i]
            com.compress()
            if com == QubitOperator('')*0.0:
                cc.append(0)
            else:
                cc.append(1)
    cc = np.array(cc)
    return np.allclose(cc,np.zeros_like(cc))

def sum_fop0_with_same_param(param,fop):
    fop0 = FermionOperator()
    for k,v in fop.terms.items():
        for kk,vv in v.items():
            if kk == param:
                fop0 += FermionOperator(k)*vv
    return fop0
def gen_ccsd_fermionic_pool(fop,pool_size):
    fermionic_pool = []
    for idx in range(pool_size):
        fop0 = sum_fop0_with_same_param('p'+str(idx),fop)
        fermionic_pool.append(fop0)
    return fermionic_pool

def split_fop_epqrs(f0):
    faaaa = FermionOperator()
    fabba = FermionOperator()
    for k,v in f0.terms.items():
        cc = []
        for kk in k:
            cc.append(kk[0])
        cc = np.array(cc)%2
        if np.allclose(cc,np.ones_like(cc)) or np.allclose(cc,np.zeros_like(cc)):
            faaaa += FermionOperator(k)*v
        else:
            fabba += FermionOperator(k)*v
    return [faaaa,fabba]
#
#  qubit pool helper
#
def exchangePauliXY(qop_k):
    """ from ((0, 'Y'), (1, 'Z'), (2, 'Z'), (3, 'Z'), (4, 'X'))
    to ((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'Z'), (4, 'Y'))
    """
    k = ()
    for item in qop_k:
        if item[1] == 'X':
            new_item = (item[0],'Y')
        elif item[1] == 'Y':
            new_item = (item[0],'X')
        elif item[1] == 'Z':
            new_item = item
        k = k + (new_item,)
    return k

def qop_exchangePauliXY(qop):
    new_qop = QubitOperator()
    for k,v in qop.terms.items():
        kk = exchangePauliXY(k)
        new_qop += QubitOperator(kk)*v
    return new_qop

def removePauliZ(qop_k):
    """ from ((0, 'Y'), (1, 'Z'), (2, 'Z'), (3, 'Z'), (4, 'X'))
    to ((0, 'Y'), (4, 'X')), hence paulistrings with maximum length 4.
    """
    k = ()
    for item in qop_k:
        if item[1] != 'Z':
            k = k+(item,) 
    return k

def qop_removePauliZ(qop):
    new_qop = QubitOperator()
    for k,v in qop.terms.items():
        kk = removePauliZ(k)
        new_qop += QubitOperator(kk)*v
    return new_qop
        
def generate_operator_pool(nqubits,nelec,pool_type,adapt_type):
    """ step1: get operator pool, qubit pool like [iP_1,iP_2,…,iP_k]
    qop come from unitary CCSD or by user defined """
    
    uccsd_dic = {'RuCCSD':RuCCSD,'RuCCGSD1':RuCCGSD1,'RuCCGSD2':RuCCGSD2,
                'UuCCSD':UuCCSD,'UuCCGSD1':UuCCGSD1,'UuCCGSD2':UuCCGSD2,
                'NuCCGSD2':NuCCGSD2,'NuCCSD':NuCCSD,'FuCCGSD2':FuCCGSD2}
    if 'Ru' in pool_type:
        t1,t2 = get_random_rccsd_type_int_t1_t2(nelec,nqubits//2)
        UCCSD = uccsd_dic[pool_type](nelec, t1, t2,'jordan_wigner')
    elif 'Uu' in pool_type:
        t1,t2 = get_random_uccsd_type_int_t1_t2(nelec,nqubits//2)
        UCCSD = uccsd_dic[pool_type](nelec, t1, t2,'jordan_wigner')
    elif 'NuCCGSD2' in pool_type:
        UCCSD = NuCCGSD2(nqubits,nelec,'jordan_wigner')
    elif 'NuCCSD' in pool_type:
        UCCSD = NuCCSD(nqubits,nelec,'jordan_wigner')
    elif 'FuCCGSD2' in pool_type:
        UCCSD = FuCCGSD2(nqubits,nelec,'jordan_wigner')
    else:
        print('NOT EXIST !')
        
    # qubit-pool
    if adapt_type == 'qubit_adapt':
        qop = UCCSD.general_qubit_generator()
        qpool = []
        for k,v in qop.terms.items(): #T-T^ = a_k*i*P_k
            # ##remove pauliZ
            kk = removePauliZ(k)
            qubit_generator = QubitOperator((kk))*1j #qubit_generator=QubitOperator((kk))*v.para_value[0]
            
            ##not remove pauliZ
            # qubit_generator = QubitOperator((k))*1j #qubit pool not remove pauliZ 20230310 ansatz removed 
            
            # qubit pool redution
            qubit_generator_ex = qop_exchangePauliXY(qubit_generator)
            if qubit_generator not in qpool and qubit_generator_ex not in qpool:
                qpool.append(qubit_generator)
        return qpool
    # fermionic_pool
    elif adapt_type == 'fermionic_adapt':
        fop = UCCSD.general_ccsd_generator()
        pool_size = UCCSD.get_amps_num()
        fermionic_pool = gen_ccsd_fermionic_pool(fop,pool_size)
        #fermionic_pool = []
        #k0 = 'p0'
        #f0 = FermionOperator()
        #for k,v in fop.terms.items():
        #    for kk,vv in v.items():
        #        if kk == k0:
        #            f0 += FermionOperator(k)
        #        else:
        #            fermionic_pool.append(f0)
        #            f0 = FermionOperator(k)
        #            k0 = kk
        #fermionic_pool.append(f0)
        #assert(len(fermionic_pool)==UCCSD.get_amps_num())    
        #　new pool split e_pqrs
        new_pool = []
        for ff in fermionic_pool:
            if ff.size == 4:
                collect = split_fop_epqrs(ff)
                new_pool = new_pool+collect
            else:
                new_pool.append(ff)
        return new_pool # fermionic_pool
    else:
        print('NOT IMPLEMRNT NOW!')
    
# ADAPT VQE imput qubit_hamiltonian and qop/qpool
# gradient = the expectation of [H, A], define a threshold for gradient norm  
def gen_pqc(nqubits,nelec,ref,opA_collect):
    ansatz_circuit = hf_input(nqubits,nelec,ref)
    for idx,pick_A in enumerate(opA_collect):
        if type(pick_A) == FermionOperator: 
            pick_A = Transform(pick_A-hermitian_conjugated(pick_A)).jordan_wigner()
        pick_A = pick_A.imag * -1
        #ansatz_unit = decompose_single_term_time_evolution(pick_A,{'p'+str(idx):1})
        ansatz_unit = TimeEvolution(pick_A*('p'+str(idx)),1.0).circuit
        ansatz_circuit += ansatz_unit
    return ansatz_circuit

def blind_pqc(adapt_info2):
    #nqubits,nelec,ref,opA_collect,theta0
    ansatz_circuit = gen_pqc(*adapt_info2[:4])
    if len(adapt_info2[3]) != 0:
        pr = dict(zip(ansatz_circuit.params_name,adapt_info2[4]))
        ansatz_circuit = ansatz_circuit.apply_value(pr=pr)
    return ansatz_circuit

def grad_measurement_obj(adapt_info2,qubit_hamiltonian,opA):
    if type(opA) == FermionOperator:
        opA = Transform(opA-hermitian_conjugated(opA)).jordan_wigner()
    qubit_HA = qubit_hamiltonian*opA
    qubit_AH = opA*qubit_hamiltonian
    commuteOP = qubit_HA - qubit_AH
    commuteOP = commuteOP.real
    
    current_circuit = blind_pqc(adapt_info2)
    grad = get_expect_value_v2(commuteOP,current_circuit)
    return np.abs(grad)

def grad_measurement(adapt_info2,qubit_hamiltonian,operator_pool,threshold,maxcycle,vqe_tol,ncore):
    grad_obj = partial(grad_measurement_obj,adapt_info2,qubit_hamiltonian)
    pool = Pool(processes=ncore)
    gnorm = pool.map(grad_obj,operator_pool)
    pool.close()
    pool.join()
    
    gnorm= np.array(gnorm)
    index = np.argmax(gnorm)
    pick_A = operator_pool[index]
    gmax = gnorm[index]
    
    if type(pick_A) == FermionOperator:
        print('check community',check_maped_op_community(pick_A))
    
    # convergence check 
    ncycle = len(adapt_info2[3])
    gconv = np.linalg.norm(gnorm) # = np.sqrt(np.sum(gnorm**2))
    if gconv < threshold:
        print('ADAPT-VQE convergence Successfull ! ncycle = {} norm_all_grad {:.9f} < threshold {}'.
              format(ncycle,gconv,threshold))
        print(gnorm)
        return True,'_',gmax,gconv
    elif gmax <= vqe_tol:
        print('ADAPT-VQE convergence Successfull ! ncycle = {} norm_opA_grad {:.9f} < vqe_tol {}'.
              format(ncycle,gmax,vqe_tol))
        print(gnorm)
        return True,'_',gmax,gconv 
    elif ncycle == maxcycle:
        print('ADAPT-VQE Terminated !upto the maxcycle= {} norm_opA_grad {:.9f} norm_all_grad {:.9f}'.
             format(maxcycle,gmax,gconv))
        return True,'_',gmax,gconv
    else:
        print('ADAPT-VQE Continue Runing ! ncycle = {} norm_all_grad {:.9f} norm_opA_grad {:.9f}'.
              format(ncycle+1,gconv,gmax))
        print('pick up operator',pick_A)
        return False,pick_A,gmax,gconv
    
def adapt_vqe(nqubits,nelec,cir_ref,qubit_hamiltonian,pool_type,ncore,adapt_type='qubit_adapt',
              threshold=1e-3,initial_type='warm_start',maxcycle=600,
              vqe_tol=1e-9,vqe_printdetails=False,vqe_savefname='No',restart_load=False):
    """ 
    Input:
    cir_ref: input the reference state usually the HF state
    qubit_hamiltonian: just qubit hamiltonian
    pool_type:'RuCCSD','RuCCGSD1','RuCCGSD2','UuCCSD','UuCCGSD1','UuCCGSD2'
    threshold: controll ADAPT convergence
    'warm_start':  The warm start initialization in which parameters are initialized in their
                   optimal values from the previous ADAPT iteration
    'cold_start': all-zero parameter initialization
    'random': uniform distribution from -pi to pi
    """
    operator_pool = generate_operator_pool(nqubits,nelec,pool_type,adapt_type)
    print('operator_pool_size:',len(operator_pool))
    print(operator_pool)
    
    if restart_load:
        opA_collect  = np.load(restart_load+'_opA.npy',allow_pickle=True).tolist()
        gmax_collect = np.load(restart_load+'_gmax.npy').tolist()
        gconv_collect= np.load(restart_load+'_gconv.npy').tolist()
        ene_collect  = np.load(restart_load+'_ene.npy').tolist()
        amp_collect  = np.load(restart_load+'_amps.npy',allow_pickle=True).tolist()
        vqe_niter    = np.load(restart_load+'_vqe_niter.npy').tolist()
        vqe_gmax     = np.load(restart_load+'_vqe_gmax.npy').tolist()
        theta0 = amp_collect[-1]
    else:
        opA_collect = []
        gmax_collect= []
        gconv_collect = []
        ene_collect = []
        amp_collect = []
        vqe_niter = []
        vqe_gmax = []
        ref = hf_input(nqubits,nelec,cir_ref)
        eref = get_expect_value_v2(qubit_hamiltonian,ref)
        ene_collect.append(eref)
        theta0 = np.array(amp_collect) # as initial

    convergence = False
    while convergence == False:
        # gradient measurement
        adapt_info2 = [nqubits,nelec,cir_ref,opA_collect,theta0]
        convergence,pick_A,gmax,gconv = grad_measurement(\
            adapt_info2,qubit_hamiltonian,operator_pool,threshold,maxcycle,vqe_tol,ncore) 
        gconv_collect.append(gconv)
        gmax_collect.append(gmax)
        # do vqe and collect information
        if convergence == False:
            opA_collect.append(pick_A)
            # set VQE initial amplitudes
            initial_amplitudes = np.zeros(len(opA_collect))
            if initial_type == 'warm_start' and len(opA_collect)>1:
                initial_amplitudes[:len(theta0)] = theta0 # warmstart
            elif initial_type == 'cold_start':
                initial_amplitudes = initial_amplitudes
            elif initial_type == 'random':
                initial_amplitudes = np.random.uniform(-np.pi,np.pi,len(ansatz_circuit.params_name))
            # run VQE, the etol in vqe should much smaller than threshold 
            ansatz_circuit = gen_pqc(nqubits,nelec,cir_ref,opA_collect)
            e0,er,theta0,_,gm,_ = get_scipy_optimized_vqe(qubit_hamiltonian,\
                                                            ansatz_circuit,\
                                                            initial_amplitudes,\
                                                            etol=vqe_tol,
                                                            printdetails=vqe_printdetails,\
                                                            savefname=vqe_savefname)
            #print('theta0',theta0)
            ene_collect.append(e0)
            amp_collect.append(theta0)
            vqe_niter.append(len(er))
            vqe_gmax.append(gm)
        
        # dump 
        filename = './'+adapt_type
        np.save(filename +'_ene',ene_collect)
        np.save(filename +'_gmax',gmax_collect)
        np.save(filename +'_gconv',gconv_collect)
        np.save(filename +'_opA',opA_collect)
        np.save(filename +'_amps',np.array(amp_collect,dtype=object))
        np.save(filename +'_vqe_niter',vqe_niter)
        np.save(filename +'_vqe_gmax',vqe_gmax)
        
    return np.array(ene_collect)

############### anlyze Tools #######################

def get_adapt_ncnot(mol,opas,amps):
    niter = len(opas)
    nc_list = [0,]
    for i in range(niter):
        info = [mol.nqubits,mol.nelec,'HF',opas[:i+1],amps[i]]
        ansatz = blind_pqc(info)
        nc = get_num_cnot(ansatz)
        nc_list.append(nc)
    return np.array(nc_list)

def get_adapt_depth(mol,opas,amps):
    niter = len(opas)
    depth_list = [1,]
    for i in range(niter):
        info = [mol.nqubits,mol.nelec,'HF',opas[:i+1],amps[i]]
        ansatz = blind_pqc(info)
        depth = get_num_depth(ansatz)
        depth_list.append(depth)
    return np.array(depth_list)

def opa_to_opakeys(opa):
    opa_new = []
    for i in range(len(opa)):
        for k,v in opa[i].terms.items():
            opa_new.append(k)
    np.save('qubit_adapt_opA_keys',np.array(opa_new,dtype=object))
    return None

def get_adapt_e0_varience(qubit_hamiltonian,bind_ansatz):
    """var = <phi_opt|(H-e0)^2|psi_opt>"""
    e0 = get_expect_value_v2(qubit_hamiltonian,bind_ansatz)
    opH = qubit_hamiltonian-e0
    opHdH = get_qubit_opVdV(opH)
    var = get_expect_value_v2(opHdH,bind_ansatz)
    return e0,var

def get_qubit_adapt_info(mol,opas,amps,niter,ref='HF'):
    adapt = np.zeros((6,niter+1))
    for i in range(niter+1):
        if i == 0:
            info = [mol.nqubits,mol.nelec,ref,[],[]] # add zero-layer info
        else:
            info = [mol.nqubits,mol.nelec,ref,opas[:i],amps[i-1]]
        ansatz = blind_pqc(info)
        
        if mol.nqubits<16:
            e0,var = get_adapt_e0_varience(mol.qubit_hamiltonian,ansatz)
        else:
            # print('hard to calculate variance, when nqubits>=16')
            e0 = get_expect_value_v2(mol.qubit_hamiltonian,ansatz)
            var = 1.0
        ss = get_expect_value_v2(mol.qubit_ss,ansatz)
        na = get_expect_value_v2(mol.qubit_na,ansatz)
        nb = get_expect_value_v2(mol.qubit_nb,ansatz)
        nalpha,nbeta = mol.nelec
        
        psi = ansatz.get_qs()
        ovlp = np.abs(psi.conj().T @ mol.vFCI)
        
        adapt[:,i] = np.array([e0,var,ovlp,ss,na-nalpha,nb-nbeta])
        #print(e0,var,ss,na,nb,1-ovlp2)
    return adapt

def get_adapt_info_v1(mol,opas,amps,niter,ref):
    
    adapt = np.zeros((9,niter+1))
    for i in range(niter+1):
        if i == 0:
            info = [mol.nqubits,mol.nelec,ref,[],[]] # add zero-layer info
        else:
            info = [mol.nqubits,mol.nelec,ref,opas[:i],amps[i-1]]
        ansatz = blind_pqc(info)
        
        if mol.nqubits<16:
            e0,var = get_adapt_e0_varience(mol.qubit_hamiltonian,ansatz)
        else:
            # print('hard to calculate variance, when nqubits>=16')
            e0 = get_expect_value_v2(mol.qubit_hamiltonian,ansatz)
            var = 1.0
        ss = get_expect_value_v2(mol.qubit_ss,ansatz)
        na = get_expect_value_v2(mol.qubit_na,ansatz)
        nb = get_expect_value_v2(mol.qubit_nb,ansatz)
        
        psi = ansatz.get_qs()
        ovlp = np.abs(psi.conj().T @ mol.vFCI)
        ovlp1 = 869527 #np.abs(psi.conj().T @ mol.vv[:,1])
        
        refA_idx,refB_idx = get_ref_idx(mol.nqubits)
        conf_ref_A = np.abs(psi[refA_idx])
        conf_ref_B = np.abs(psi[refB_idx])
        
        adapt[:,i] = np.array([e0,var,ovlp,ovlp1,ss,na,nb,conf_ref_A,conf_ref_B])
        #print(e0,var,ss,na,nb,1-ovlp2)
    return adapt

def ana_qubit_adapt(mol1,ref,niter=None):
    amps = np.load('./qubit_adapt_amps.npy',allow_pickle=True)
    opas = np.load('./qubit_adapt_opA.npy',allow_pickle=True)
    # depth
    depth = get_adapt_depth(mol1,opas,amps)
    np.save('./qubit_adapt_depth',depth)
    # ncnot
    ncnot = get_adapt_ncnot(mol1,opas,amps)
    np.save('./qubit_adapt_ncnot',ncnot)
    
    if niter is None:
        niter = len(opas)
    adapt_info = get_adapt_info_v1(mol1,opas,amps,niter,ref)
    np.save('./adapt_info.npy',adapt_info)
    #　
    adapt = np.zeros((1,13,niter+1))
    adapt[0,0,:] = adapt_info[0,:]
    adapt[0,1,:] = adapt_info[1,:]
    adapt[0,2,:] = adapt_info[2,:]
    adapt[0,3,:] = adapt_info[3,:]
    adapt[0,6,:] = adapt_info[4,:]
    adapt[0,7,:] = adapt_info[5,:]
    adapt[0,8,:] = adapt_info[6,:]
    adapt[0,10,:] = adapt_info[7,:]
    adapt[0,11,:] = adapt_info[8,:]
    np.save('../../analyze/Qubit_ADAPT',adapt)
    return None

def ana_fermionic_adapt(mol1,ref,niter=None):
    amps = np.load('./fermionic_adapt_amps.npy',allow_pickle=True)
    opas = np.load('./fermionic_adapt_opA.npy',allow_pickle=True)
    # depth
    depth = get_adapt_depth(mol1,opas,amps)
    np.save('./fermionic_adapt_depth',depth)
    # ncnot
    ncnot = get_adapt_ncnot(mol1,opas,amps)
    np.save('./fermionic_adapt_ncnot',ncnot)
    if niter is None:
        niter = len(opas)
    adapt_info = get_adapt_info_v1(mol1,opas,amps,niter,ref)
    np.save('./adapt_info.npy',adapt_info)
    #　
    adapt = np.zeros((1,13,niter+1))
    adapt[0,0,:] = adapt_info[0,:]
    adapt[0,1,:] = adapt_info[1,:]
    adapt[0,2,:] = adapt_info[2,:]
    adapt[0,3,:] = adapt_info[3,:]
    adapt[0,6,:] = adapt_info[4,:]
    adapt[0,7,:] = adapt_info[5,:]
    adapt[0,8,:] = adapt_info[6,:]
    adapt[0,10,:] = adapt_info[7,:]
    adapt[0,11,:] = adapt_info[8,:]
    np.save('../../analyze/Fermionic_ADAPT',adapt)
    return None
