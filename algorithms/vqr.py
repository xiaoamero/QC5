import time
import numpy as np
import h5py
import scipy.optimize
from .grad_ops_wrapper import *

def fun1(ansatz_amps, xAdAx_pqc):
    f, g = xAdAx_pqc(ansatz_amps)
    f = np.real(f)[0, 0]
    g = np.real(g)[0, 0]
    return f, g

def fun2(ansatz_amps, encoder_amps, VdA_real_pqc, VdA_imag_pqc):
    fr, _, gr = VdA_real_pqc(encoder_amps, ansatz_amps)
    fr = fr[0, 0]
    gr = gr[0, 0]
    
    fi, _, gi = VdA_imag_pqc(encoder_amps, ansatz_amps)
    fi = fi[0, 0]
    gi = gi[0, 0]
 
    fri = fr+1j*fi
    f = np.abs(fri)**2 
    gri = gr+1j*gi
    g = 2*np.real(fri*np.conj(gri))
    
    return f, g

def energy_objective_vqr(ansatz_amps,encoder_amps,
                         ozzo,
                         xAdAx_pqc,
                         VdA_real_pqc,VdA_imag_pqc):

    f1, g1 = fun1(ansatz_amps, xAdAx_pqc)
    f2, g2 = fun2(ansatz_amps, encoder_amps, VdA_real_pqc, VdA_imag_pqc)
    f = ozzo*f1 - f2
    g = ozzo*g1 - g2
    
    return f, g

def check_convergence(iter_count,iter_recoder,maxcycle,etol):
    if iter_count <= 1:
        print('WARNING: Total iterative number:',iter_count,' Please Check If It Is Normal !')
    elif iter_count >= 2:
        xmax = max(np.abs(iter_recoder[-2]),np.abs(iter_recoder[-1]),1)
        conv_ok = (iter_recoder[-2] - iter_recoder[-1])/xmax
        if conv_ok < etol:
            print('Convergence Successful ! For brevity of the output,here, we only print the first ten iterations.')
            print('Total iterative number:',iter_count,', f(k)-f(k-1)=',conv_ok)
        else:
            print('Convergence Failure !')
            print('Total iterative number:',iter_count,' maxcycle:',maxcycle,' f(k)-f(k-1)=',conv_ok)
    return None

def get_scipy_optimized_vqr(initial_amplitudes,encoder_amps,
                            oVdVo,xAdAx_pqc,VdA_real_pqc,VdA_imag_pqc,
                            optimizer,etol,savefile):     
    vqr_csf_recoder = []
    vqr_amp_recoder = []    
    #print("VQR initial amplitudes", len(initial_amplitudes),initial_amplitudes.tolist())
    print("VQR initial amplitudes", len(initial_amplitudes))
    print("VQR Runing under the "+ optimizer + " optimizer with threshold",etol)
    global count
    count = 0   
    def callback_fn(amplitudes):
        global count
        cost_func,_ = energy_objective_vqr(amplitudes,encoder_amps,
                                           oVdVo,xAdAx_pqc,
                                           VdA_real_pqc,VdA_imag_pqc)
        vqr_csf_recoder.append(cost_func)
        vqr_amp_recoder.append(amplitudes)
        if count <= 10:
            print("Time: {}; Iteration {} cost_func = {}".format(time.ctime(),count,cost_func))
        count +=1
        return None    
    callback_fn(initial_amplitudes)
    
    start_time = time.time()
    result = scipy.optimize.minimize(energy_objective_vqr,
                                     initial_amplitudes,
                                     args=(encoder_amps,oVdVo,xAdAx_pqc,VdA_real_pqc,VdA_imag_pqc),
                                     method=optimizer,
                                     jac=True,
                                     tol=etol,
                                     callback=callback_fn)    
    cost_fun = result.fun
    opted_amps = result.x
    end_time = time.time()
    vqr_time = round(end_time-start_time,3)
    # check convergence
    print('For brevity of the output, here, we only print the first ten iterations.')
    maxcycle = len(result.x)*200 # by scipy default
    check_convergence(count-1,vqr_csf_recoder,maxcycle,etol)
    print('Total time',vqr_time,'s')
    # save iter data
    if savefile:
        file_name_index = str(int(time.time()))[-6:]
        print('vqr_file_name_index',file_name_index)
        np.save('./data/vqr_csf_recoder'+file_name_index,vqr_csf_recoder)
        np.save('./data/vqr_amp_recoder'+file_name_index,vqr_amp_recoder)
        
    return cost_fun,opted_amps,vqr_csf_recoder

def get_chi_omega(omega, gamma, vqr_initial_amps,
                  vqr_ansatz_circuit,vqe_ansatz_circuit,
                  theta0, e0, qubit_hamiltonian, opV,
                 optimizer='L-BFGS-B',etol=1e-10,savefile=False):
    print('omega=',omega,'gamma=',gamma)
    opVdV = get_qubit_opVdV(opV)   
    opAdA,opA,opAd,opAd_real,opAd_imag = get_qubit_opA(qubit_hamiltonian,e0,omega,gamma)
    opVdA,opVdA_real,opVdA_imag = get_qubit_opVdA(opV,opA)
    
    oVdVo = get_expect_value(opVdV,vqe_ansatz_circuit,theta0)   
    xAdAx_pqc    = get_expect_pqc(opAdA,vqr_ansatz_circuit)
    VdA_real_pqc = get_cross_pqc(opVdA_real,vqr_ansatz_circuit,vqe_ansatz_circuit)
    VdA_imag_pqc = get_cross_pqc(opVdA_imag,vqr_ansatz_circuit,vqe_ansatz_circuit)
    # VQR STEP1
    encoder_amps = theta0.reshape(1,theta0.shape[0])
    cost_fun,theta1,vqr_csf_recoder = get_scipy_optimized_vqr(vqr_initial_amps,
                                                              encoder_amps,
                                                              oVdVo,
                                                              xAdAx_pqc,
                                                              VdA_real_pqc,
                                                              VdA_imag_pqc,
                                                              optimizer=optimizer,
                                                              etol=etol,
                                                              savefile=savefile)
    amps_ovlp = get_ovlp_value(vqr_initial_amps,theta1,vqr_ansatz_circuit)
    print('VQR optimized amplitudes:',amps_ovlp**2, len(theta1), theta1.tolist())
    #print('VQR optimized amplitudes:',amps_ovlp**2, len(theta1))
    print('VQR cost function:',cost_fun)
    
    # VQR STEP2
    xAdAx = get_expect_value(opAdA,vqr_ansatz_circuit,theta1)
    mode_square = oVdVo/xAdAx    
    xAdx_real = get_expect_value(opAd_real,vqr_ansatz_circuit,theta1)
    xAdx_imag = opAd_imag 
    real = xAdx_real*mode_square
    imag = xAdx_imag*mode_square        
    chi = real+1j*imag
    print("Chi(omega): {} for omega = {} gamma = {}".format(chi,omega,gamma))
    return chi,theta1,cost_fun

def get_chi_omega_value(omega, gamma, theta1,
                  vqr_ansatz_circuit,vqe_ansatz_circuit,
                  theta0, e0, qubit_hamiltonian, opV):
    # print('omega=',omega,'gamma=',gamma)
    opVdV = get_qubit_opVdV(opV)   
    opAdA,opA,opAd,opAd_real,opAd_imag = get_qubit_opA(qubit_hamiltonian,e0,omega,gamma)
    oVdVo = get_expect_value(opVdV,vqe_ansatz_circuit,theta0)   
    # VQR STEP1    
    # VQR STEP2
    xAdAx = get_expect_value(opAdA,vqr_ansatz_circuit,theta1)
    mode_square = oVdVo/xAdAx    
    xAdx_real = get_expect_value(opAd_real,vqr_ansatz_circuit,theta1)
    xAdx_imag = opAd_imag 
    real = xAdx_real*mode_square
    imag = xAdx_imag*mode_square        
    chi = real+1j*imag
    return chi