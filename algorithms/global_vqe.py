import numpy as np
import h5py
import time
import os
from functools import partial
from multiprocessing import Pool

from mindquantum.core.operators import QubitOperator
from QC5.ansatze.hea_and_hva.get_cir import *
from .grad_ops_wrapper import get_expect_value_v2
from .vqe import get_scipy_optimized_vqe

# global search VQE

def empty_params_ndarray(nqubits,nlayer,hea_type,size):
    """return empty ndarray for dump different layer params"""
    nparams = get_hea_nparams(nqubits,nlayer,hea_type,size)
    empty_amps_nadrray = np.zeros((nlayer,nparams)) 
    return empty_amps_nadrray

def write_h5py(nqubits,nlayer,hea_type,size,npoints,filename,restart_layer):
    nlayer0 = nlayer+1 # +1 for dump ref information
    energy_forward    = np.zeros(nlayer0)
    time_forward      = np.zeros(nlayer0)
    global_points     = np.zeros((nlayer0,npoints))
    amps_forward      = empty_params_ndarray(nqubits,nlayer0,hea_type,size)
    niter_forward     = np.zeros(nlayer0)
    max_gnorm_forward = np.zeros(nlayer0)
    # write                  
    fname = filename+'_forward.h5'
    f = h5py.File(fname,'w')
    f['collect_energy']    = energy_forward   
    f['collect_time']      = time_forward     
    f['collect_points']    = global_points    
    f['collect_amps']      = amps_forward     
    f['collect_niter']     = niter_forward    
    f['collect_max_gnorm'] = max_gnorm_forward
    f.close()
    
    if restart_layer==1:
        pre_layers_amps = []
        return energy_forward,time_forward,global_points,\
               amps_forward,niter_forward,max_gnorm_forward,pre_layers_amps
    else:
        assert(restart_layer>1)
        print('Global vqe go on ')
        return combination(restart_layer,nlayer,hea_type,nqubits,size)

def update_h5py(filename,energy_forward,time_forward,global_points,
                amps_forward,niter_forward,max_gnorm_forward):
    fname = filename+'_forward.h5'
    f = h5py.File(fname,'a')
    f['collect_energy'][:] = energy_forward
    f['collect_time'][:] = time_forward
    f['collect_points'][:] = global_points
    f['collect_amps'][:] = amps_forward
    f['collect_niter'][:] = niter_forward
    f['collect_max_gnorm'][:] = max_gnorm_forward
    f.close()
    return None

def combination(restart_layer,nlayer,hea_type,nqubits,size):
    pre_nlayer = restart_layer-1
    new_nlayer = nlayer
    pre_fname = hea_type+'_nlayer'+str(pre_nlayer)+'_forward.h5'
    f = h5py.File(pre_fname,'r')
    pre_energy_forward = f['collect_energy'][()]
    pre_time_forward = f['collect_time'][()]
    pre_global_points  = f['collect_points'][()]
    pre_amps_forward   = f['collect_amps'][()]  
    pre_niter_forward  = f['collect_niter'][()] 
    pre_max_gnorm_forward = f['collect_max_gnorm'][()]
    f.close()

    new_fname = hea_type+'_nlayer'+str(new_nlayer)+'_forward.h5'
    f = h5py.File(new_fname,'r')
    new_energy_forward = f['collect_energy'][()]
    new_time_forward = f['collect_time'][()]
    new_global_points  = f['collect_points'][()]
    new_amps_forward   = f['collect_amps'][()]  
    new_niter_forward  = f['collect_niter'][()] 
    new_max_gnorm_forward = f['collect_max_gnorm'][()]
    f.close()
    
    # combination
    for idx in range(pre_nlayer+1):
        new_energy_forward[idx] = pre_energy_forward[idx] 
        new_time_forward[idx] = pre_time_forward[idx]
        new_global_points[idx] = pre_global_points[idx]
        amin= min(len(pre_amps_forward[pre_nlayer]),len(new_amps_forward[-1]))
        new_amps_forward[idx][:amin] = pre_amps_forward[idx][:amin]      
        new_niter_forward[idx] = pre_niter_forward[idx]
        new_max_gnorm_forward[idx]= pre_max_gnorm_forward[idx]
        
    nparams = get_hea_nparams(nqubits,pre_nlayer,hea_type,size)
    pre_layers_amps = pre_amps_forward[pre_nlayer][:nparams]
    return new_energy_forward,new_time_forward,new_global_points,\
           new_amps_forward,new_niter_forward,new_max_gnorm_forward,\
           pre_layers_amps.tolist()

def gen_1layer_new_amps_pool(rng,nparams,ndirection,stepsizelist,pre_layers_amps):
    nparams1 = nparams - len(pre_layers_amps)
    pool_info_list = []
    zeros = pre_layers_amps+[0,]*nparams1
    pool_info_list.append((0,0,np.array(zeros)))
    print('Generating initial guess, direction:{}, stepsize:{}, largest amplitudes:{}'.format(0,0,0))
    for ndx in range(1,ndirection+1):
        amps1 = rng.uniform(-1,1,nparams1)
        amps1 = amps1/np.amax(np.abs(amps1))
        for sdx in stepsizelist:
            stepsize1 = 2*np.pi/2**sdx
            amps1_ = amps1*stepsize1
            randoms = pre_layers_amps+amps1_.tolist()
            pool_info_list.append((ndx,sdx,np.array(randoms)))
            print('Generating initial guess, direction:{}, stepsize:{}, largest amplitudes:{}'\
                  .format(ndx,sdx,stepsize1))
    return pool_info_list

def gen_nlayer_new_amps_pool(rng,nparams,ndirection,stepsizelist,restart_layer,nth,initial_amps):
    # initial_amps from cascade, qubit_adapt ~
    pool_info_list = []
    from_cascade = initial_amps[nth-restart_layer][:nparams] # nth-2 start from 0
    pool_info_list.append((0,0,from_cascade))
    print('Generating initial guess, direction:{}, stepsize:{}, largest amplitudes:{}'.format(0,0,0))
    for ndx in range(1,ndirection+1):
        amps1 = rng.uniform(-1,1,nparams)
        amps1 = amps1/np.amax(np.abs(amps1))   
        for sdx in stepsizelist:
            stepsize1 = 2*np.pi/2**sdx
            amps1_ = amps1*stepsize1
            randoms = from_cascade + amps1_
            pool_info_list.append((ndx,sdx,randoms))
            print('Generating initial guess, direction:{}, stepsize:{}, largest amplitudes:{}'\
                  .format(ndx,sdx,stepsize1))
    return pool_info_list

def vqe_obj_simple(vqe_tol,vqe_niter,qubit_hamiltonian,hea_info,pool_info):
    # hea_info = [nqubits,nlayer,hea_type,size,nelec,ref]
    ansatz_circuit = get_hea_ansatz(*hea_info)
    ndx,sdx,initial_amplitudes = pool_info
    e0,ene_recoder,theta0,amps_recoder,\
    max_grad_norm,max_gnorm_recoder = get_scipy_optimized_vqe(
        qubit_hamiltonian,ansatz_circuit,initial_amplitudes,
        optimizer='BFGS',etol=vqe_tol,maxniter=vqe_niter,printdetails=False) 
    niter = len(ene_recoder)-1
    recoder = [ene_recoder,[amps_recoder[0],amps_recoder[-1]],max_gnorm_recoder]
    return e0,theta0,niter,max_grad_norm,ndx,sdx,recoder

def save_emin_vqe(nthlayer,ndx,sdx,recoder):
    vqefname = 'vqe/layer'+str(nthlayer)+'_direction'+str(ndx)+'_stepsize'+str(sdx)
    vqe_ene_recoder,vqe_amp_recoder,vqe_max_gnorm_recoder = recoder
    f = h5py.File(vqefname + '.h5','w')
    f['ene_recoder'] = vqe_ene_recoder
    f['amps_recoder'] = vqe_amp_recoder
    f['max_gnorm_recoder'] = vqe_max_gnorm_recoder
    f.close() 
    return None

def GlobalVQE(qubit_hamiltonian,nqubits,nlayer,hea_type,
              size=(0,0),nelec=(0,0),hea_ref='',
              restart_layer=1,
              ndirection=1,stepsizelist=[0],ncore=2,seed=None,
              vqe_tol = 1e-6,vqe_niter='default',
              initial_amps='layerwise'):
    """
    a. restart calculate, you should provide restart_layer and previous-layers opted amps (list), 
    b. from cascade to YXY2F the restart_layer should = 2 !
    """
    time1 = time.time()
    if not os.path.exists('vqe'):
        os.makedirs('vqe')
    print('See the "vqe" folder for details')
    print('nqubits',nqubits,'nelec',nelec)
        
    # ----check data type-------
    if type(stepsizelist) == list:
        stepsizelist = np.array(stepsizelist)
    # ---------write or load -------------
    filename = hea_type+'_nlayer'+str(nlayer)
    nsearch = len(stepsizelist)*ndirection+1
    energy_forward,time_forward,global_points,amps_forward,\
          niter_forward,max_gnorm_forward,pre_layers_amps=\
          write_h5py(nqubits,nlayer,hea_type,size,nsearch,filename,restart_layer)
    # -------ref energy---------
    if restart_layer == 1:
        cir_ref = hf_input(nqubits,nelec,hea_ref)
        eref = get_expect_value_v2(qubit_hamiltonian,cir_ref)
        energy_forward[0] = eref
    # ---------Calculate ---------
    rng = np.random.default_rng(seed)
    for nth in range(restart_layer,nlayer+1):
        # prepare amplitudes
        nparams = get_hea_nparams(nqubits,nth,hea_type,size)
        # ------ global search -----------
        if type(initial_amps) == str:
            pool_info_list = gen_1layer_new_amps_pool(rng,nparams,
                                                      ndirection,stepsizelist,
                                                      pre_layers_amps)     
        elif type(initial_amps) == np.ndarray:
            # using for cascade to YXY2F, restart_layer should set as 2 and input initial_amps
            # using for adapt-vqe to XYZ1F or XYZ2F, restart_layer should set as 1 and input initial_amps
            pool_info_list = gen_nlayer_new_amps_pool(rng,nparams,
                                                      ndirection,stepsizelist,
                                                      restart_layer,nth,initial_amps)   
        # ---------parallel---------------      
        hea_info = [nqubits,nth,hea_type,size,nelec,hea_ref]
        vqe_obj = partial(vqe_obj_simple,vqe_tol,vqe_niter,qubit_hamiltonian,hea_info)
        # pool
        st = time.time()
        pool = Pool(processes=ncore)
        res = pool.map(vqe_obj,pool_info_list)
        pool.close()
        pool.join()
        et = time.time()
        tot = et-st
        print('Runing Multiprocess, layer {} time consuming: {:.3f} s ncore={}'.format(nth,tot,ncore))
        # ---------post analyze------------
        collect_e = [i[0] for i in res]
        min_idx = collect_e.index(min(collect_e))
        print('layer {} min_idx {} ndirection {} stepsize {}'.format(nth,min_idx,res[min_idx][4],res[min_idx][5]))
        # ---------update data-------------
        global_points[nth] = collect_e
        energy_forward[nth] = res[min_idx][0]
        time_forward[nth] = tot
        amps_forward[nth][:nparams] = res[min_idx][1]
        niter_forward[nth] = res[min_idx][2]
        max_gnorm_forward[nth] = res[min_idx][3]
        update_h5py(filename,energy_forward,time_forward,global_points,\
                    amps_forward,niter_forward,max_gnorm_forward)
        save_emin_vqe(nth,res[min_idx][4],res[min_idx][5],res[min_idx][6])
        # --------pre_layers_amps--------
        pre_layers_amps = res[min_idx][1].tolist() # min_theta0 = res[min_idx][1]
        np.save('elst',energy_forward)
        print('================================================================')
    time2 = time.time()
    total = time2-time1
    print('Total time:',total,'s')
    return energy_forward #,pre_layers_amps