"""
   Generate spin restricted unitary coupled cluster singles and doubles ansatz,
   where singles are generalized.
   
"""
import numpy
from mindquantum.core.operators import TimeEvolution,FermionOperator
from mindquantum.core.operators.utils import hermitian_conjugated
from mindquantum.algorithm.nisq.chem.transform import Transform
from mindquantum import Circuit, X
from .UnitaryCCSD_base import UnitaryCCSD

class RuCCGSD1complex(UnitaryCCSD):
    
    def __init__(self, nelec, t1, t2, fermion_transform):
        super().__init__(nelec, t1, t2, fermion_transform)
    
    def get_amps_num(self):
        no,nv = self.t1.shape
        nov = no*nv
        nmo = no+nv
        num = nmo*(nmo-1)//2 + nov + nov*(nov-1)//2
        complex_num = num*2
        return complex_num
    
    def get_packed_amps(self):
        no,nv = self.t1.shape
        t1 = self.t1
        t2 = self.t2
    
        t1_list = []
        t2_list = []
        nmo = no + nv
        # t1_list
        for p in range(nmo):        
            for q in range(p):
                if p >= no and q < no:
                    us = t1[q,p-no]
                else:
                    us = numpy.random.uniform(-1,1)*self.parameter
                t1_list.append(us)
                           
        # t2_list
        for i in range(no):
            for a in range(nv):
                for j in range(no):
                    for b in range(nv):
                        ia = i*nmo + a
                        jb = j*nmo + b
                        if ia>=jb:
                            ud = t2[i,j,a,b]
                            t2_list.append(ud)
        
        packed_amps = (t1_list + t2_list)*2 # repeat list,imag = real
        packed_amps = numpy.array(packed_amps)
        packed_amps[abs(packed_amps) < 1e-8] = 0
        return packed_amps
    
    def general_ccsd_generator(self,param_expression=True):
        """ return the fermion_generator of corresponding ansatz,
            packed_amps = ["p1", "p2", "p3", ...] """
        no,nv = self.t1.shape
        nmo = no + nv
        amps_num = self.get_amps_num()
        med = amps_num//2
        if param_expression:
            packed_amps = ["p" + str(i) for i in range(amps_num)]
        else:
            packed_amps = self.get_packed_amps()
        # t1
        idx = 0
        su = FermionOperator() 
        for p in range(nmo):
            for q in range(p):
                t1re = packed_amps[idx] 
                t1im = packed_amps[idx+med]
                
                orb_pA = 2*p
                orb_qA = 2*q
                t1A = FermionOperator(((orb_pA,1),(orb_qA,0)),t1re)
                t1A += FermionOperator(((orb_pA,1),(orb_qA,0)),t1im)*1j
                
                orb_pB = 2*p+1
                orb_qB = 2*q+1
                t1B = FermionOperator(((orb_pB,1),(orb_qB,0)),t1re)
                t1B += FermionOperator(((orb_pB,1),(orb_qB,0)),t1im)*1j

                su += t1A + t1B
                idx += 1  
        # t2
        du = FermionOperator()
        for i in range(no):
            for a in range(nv):
                for j in range(no):
                    for b in range(nv):
                        ia = i*nmo + a
                        jb = j*nmo + b
                        if ia>=jb:    
                            t2re = packed_amps[idx]
                            t2im = packed_amps[idx+med]
                            if ia>jb: 
                                orb_iA = 2*i
                                orb_aA = 2*(a+no)
                                orb_jA = 2*j
                                orb_bA = 2*(b+no)
                                t2AA = FermionOperator(((orb_aA,1),(orb_bA,1),(orb_jA,0),(orb_iA,0)),t2re)
                                t2AA += FermionOperator(((orb_aA,1),(orb_bA,1),(orb_jA,0),(orb_iA,0)),t2im)*1j
                                
                                orb_iA = 2*i
                                orb_aA = 2*(a+no)
                                orb_jB = 2*j+1
                                orb_bB = 2*(b+no)+1
                                t2AB = FermionOperator(((orb_aA,1),(orb_bB,1),(orb_jB,0),(orb_iA,0)),t2re)
                                t2AB += FermionOperator(((orb_aA,1),(orb_bB,1),(orb_jB,0),(orb_iA,0)),t2im)*1j
                                
                                orb_iB = 2*i+1
                                orb_aB = 2*(a+no)+1
                                orb_jA = 2*j
                                orb_bA = 2*(b+no)
                                t2BA = FermionOperator(((orb_bA,1),(orb_aB,1),(orb_iB,0),(orb_jA,0)),t2re)
                                t2BA += FermionOperator(((orb_bA,1),(orb_aB,1),(orb_iB,0),(orb_jA,0)),t2im)*1j
                                
                                orb_iB = 2*i+1
                                orb_aB = 2*(a+no)+1
                                orb_jB = 2*j+1
                                orb_bB = 2*(b+no)+1
                                t2BB = FermionOperator(((orb_aB,1),(orb_bB,1),(orb_jB,0),(orb_iB,0)),t2re)
                                t2BB += FermionOperator(((orb_aB,1),(orb_bB,1),(orb_jB,0),(orb_iB,0)),t2im)*1j
                                
                                du += t2AA+t2AB+t2BA+t2BB
                            else:
                                orb_iA = 2*i
                                orb_aA = 2*(a+no)
                                orb_jB = 2*j+1
                                orb_bB = 2*(b+no)+1
                                t2AB = FermionOperator(((orb_aA,1),(orb_bB,1),(orb_jB,0),(orb_iA,0)),t2re)
                                t2AB += FermionOperator(((orb_aA,1),(orb_bB,1),(orb_jB,0),(orb_iA,0)),t2im)*1j
                                du += t2AB                    
                            idx += 1
        assert(2*idx==len(packed_amps))
        hat_T = su + du
        return hat_T