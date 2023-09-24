import numpy as np 
import matplotlib.pyplot as plt
from .get_cir import *

def gen_x_axis(size,max_layer,hea_type,set_x='L'):
    nsite = size[0]*size[1]
    nqubits = 2*nsite
    
    if set_x == 'L':
        return range(max_layer)
    elif set_x == 'nparams':
        nparams = []
        for nth in range(max_layer):
            n0 = get_hea_nparams(nqubits, nth, hea_type,size=size)
            nparams.append(n0)
        return nparams
    elif set_x == 'CNOT':
        ncnot = []
        for nth in range(max_layer):
            if 'hva' in hea_type:
                nref = 2*(nsite-nsite//2)*nsite//2
            else:
                nref = 0
            n0 = cal_ncnot(nqubits, nth, hea_type, size,nref=0)+nref
            ncnot.append(n0)
        return ncnot
    elif set_x == 'Depth':
        ndepth = []
        for nth in range(max_layer):
            if 'hva' in hea_type:
                nref = nsite-1
            else:
                nref = 1
            n0 = cal_ndepth(nqubits, nth, hea_type, size, nref=0)+nref
            ndepth.append(n0)
        return ndepth

def error_plot(data,size,efci,xtype='L',sp={}):
    if sp =={}:
        sp = {
          'ry_cascade':'C0o-','ry_full': 'C4o-','ry_linear': 'C6o-', 
          'EfficientSU2':'C2o-', 'XYZ2F':'C1o-','ASWAP':'C5o-', 
          'fSim_linear':'C7o-','fSim_brickwall':'C8o-',
          'XYZ2F_2D_hubbard':'C3s--',
          'ry_cascade_2DX_hubbard':'C0s--','XYZ2F_2DX_hubbard':'C1s--',
          'ASWAP_2DX_hubbard':'C5s--',
          'fSim_linear_2DX_hubbard':'C7s--','fSim_brickwall_2DX_hubbard':'C8s--',
          'hva_hubbard':'C9s--'}
    
    fig = plt.figure(figsize=(7,5))
    wordsize = 16
    plt.tick_params(labelsize=wordsize)
    plt.grid()
    
    for k in data.keys():
        hea_type = k
        y = data[k]
        max_layer=len(y)
        x = gen_x_axis(size,max_layer,hea_type,set_x=xtype)
        plt.plot(x,(y-efci),sp[hea_type],label=hea_type)
    
    plt.yscale('log')
    plt.yticks(np.logspace(-10, 10, num=21))
    plt.ylim(1e-5,1e1)
    if xtype == 'L':
        plt.xlabel('$L$',fontsize=wordsize)
    elif xtype == 'nparams':
        plt.xscale('log')
        plt.xlabel('Number of parameters',fontsize=wordsize)
    elif xtype == 'CNOT':
        plt.xscale('log')
        plt.xlabel('Number of 2-Qubit gates',fontsize=wordsize) 
    elif xtype == 'Depth':
        plt.xscale('log')
        plt.xlabel('Circuit Depth',fontsize=wordsize)
    plt.ylabel('Error (a.u.)',fontsize=wordsize)
    plt.title('size='+str(size)+', PBC = False',fontsize=wordsize)
    plt.legend(bbox_to_anchor=(1.0, 1.0),loc='upper left',borderaxespad=0.1,frameon=False,fontsize=wordsize)
    plt.show()
    return None