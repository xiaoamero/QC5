import time
import numpy as np
import h5py
import scipy.optimize
from .grad_ops_wrapper import *

def energy_objective_vqe(ansatz_amps,ene_pqc_op):
    f, g = ene_pqc_op(ansatz_amps)
    f = np.real(f)[0, 0]
    g = np.real(g)[0, 0]
    return f,g
    
def get_scipy_optimized_vqe(qubit_hamiltonian,vqe_ansatz_circuit,initial_amplitudes,
                            optimizer='BFGS',etol=1e-6,maxniter='default',
                            printdetails=False,savefname='No'):
    vqe_ene_recoder = []
    vqe_amp_recoder = []
    vqe_max_gnorm_recoder = []
    nparams = len(initial_amplitudes)
    ene_pqc  = get_expect_pqc(qubit_hamiltonian,vqe_ansatz_circuit)
    print("VQE Runing under the "+ optimizer + " optimizer with threshold",etol,'maxniter',maxniter)
    
    global count
    count = 0   
    def callback_fn(amplitudes,printiter=printdetails):
        global count
        energy,grad = energy_objective_vqe(amplitudes,ene_pqc)
        max_gnorm = np.amax(np.abs(grad))
        vqe_ene_recoder.append(energy)
        vqe_amp_recoder.append(amplitudes)
        vqe_max_gnorm_recoder.append(max_gnorm)
        if printiter:
            print("Time: {}; Iteration {} expectation = {}".format(time.ctime(),count,energy))
        count +=1
        return None
    callback_fn(initial_amplitudes)  # iter0 info
    
    if type(maxniter) == int:
        niter = maxniter
    else:
        niter = 200*nparams # scipy default
    
    start_time = time.time()
    result = scipy.optimize.minimize(energy_objective_vqe,
                                     initial_amplitudes,
                                     args=(ene_pqc),
                                     method=optimizer,
                                     jac=True,
                                     tol=etol,
                                     callback=callback_fn,
                                     options={'maxiter':niter,'disp': True}) # options={'maxiter': 2,'disp': True}
    
    # print VQE message
    print('-----------------  VQE Information Summary  -----------------')
    energy = result.fun
    opted_amps = result.x
    grad = result.jac
    max_gnorm = np.amax(np.abs(grad))
    end_time = time.time()
    vqe_time = round(end_time-start_time,3)
    
    print('Success:',result.success)
    print(result.message)
    print('Energy: {} \nAmplitudes number: {} \nTime Consuming: {} s \
         \nIteration Number: {} \nMax_gnorm: {}'.format(energy,nparams,vqe_time,count-1,max_gnorm))
    if printdetails:
        amps_ovlp = get_ovlp_value(initial_amplitudes,opted_amps,vqe_ansatz_circuit)
        print('Overlap between psi_0 and psi_opt is {} nparams is {}'.format(amps_ovlp**2,nparams))
        print('Initial Amplitudes: \n{}'.format(initial_amplitudes.tolist()))
        print('Optimized Amplitudes: \n{}'.format(opted_amps.tolist()))
    print('--------------------------------------------------------------')
    if savefname != 'No':
        f = h5py.File(savefname + '.h5','w')
        f['ene_recoder'] = vqe_ene_recoder
        f['amps_recoder'] = [vqe_amp_recoder[0],vqe_amp_recoder[-1]]
        f['max_gnorm_recoder'] = vqe_max_gnorm_recoder
        f['time_recoder'] = vqe_time
        f.close()    
    return energy,vqe_ene_recoder,opted_amps,vqe_amp_recoder,max_gnorm,vqe_max_gnorm_recoder 