"""
   Generate spin restricted unitary coupled cluster generalized singles and doubles ansatz.
"""
import numpy
from mindquantum.core.operators import TimeEvolution,FermionOperator
from mindquantum.core.operators.utils import hermitian_conjugated
from mindquantum.algorithm.nisq.chem.transform import Transform
from mindquantum import Circuit, X
from .UnitaryCCSD_base import UnitaryCCSD

class RuCCGSD2(UnitaryCCSD):
    
    def __init__(self, nelec, t1, t2, fermion_transform):
        super().__init__(nelec, t1, t2, fermion_transform)
        
    def get_amps_num(self):
        no,nv = self.t1.shape
        nmo = no + nv
        # t1_list
        num = nmo*(nmo-1)//2
        # t2_list
        for p in range(nmo):
            for q in range(nmo):
                for r in range(nmo):
                    for s in range(nmo):
                        pr = p*nmo + r
                        qs = q*nmo + s
                        pq = p*nmo + q
                        rs = r*nmo + s
                        if pr>=qs and rs > pq:
                            num += 1
        return num
    
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
        for p in range(nmo):
            for q in range(nmo):
                for r in range(nmo):
                    for s in range(nmo):
                        pr = p*nmo + r
                        qs = q*nmo + s
                        pq = p*nmo + q
                        rs = r*nmo + s
                        if pr>=qs and rs > pq:
                            if p < no and q < no and s >= no and r>=no:
                                ud = t2[p,q,r-no,s-no]
                            else:
                                ud = numpy.random.uniform(-1,1)*self.parameter
                            t2_list.append(ud)
        packed_amps = t1_list + t2_list
        packed_amps = numpy.array(packed_amps)
        packed_amps[abs(packed_amps) < 1e-8] = 0
        return packed_amps
    
    def general_ccsd_generator(self,param_expression=True):
        """ return the fermion_generator of corresponding ansatz,
            packed_amps = ["p1", "p2", "p3", ...] """
        no,nv = self.t1.shape
        nmo = no + nv
        amps_num = self.get_amps_num()
        if param_expression:
            packed_amps = ["p" + str(i) for i in range(amps_num)]
        else:
            packed_amps = self.get_packed_amps()
        # t1
        su = FermionOperator() 
        idx = 0
        for p in range(nmo):
            for q in range(p):
                t1x = packed_amps[idx]
                
                orb_pA = 2*p
                orb_qA = 2*q
                t1A = FermionOperator(((orb_pA,1),(orb_qA,0)),t1x)
                
                orb_pB = 2*p+1
                orb_qB = 2*q+1
                t1B = FermionOperator(((orb_pB,1),(orb_qB,0)),t1x)
                
                su += t1A + t1B
                idx += 1     
        # t2
        du = FermionOperator() 
        for p in range(nmo):
            for q in range(nmo):
                for r in range(nmo):
                    for s in range(nmo):
                        pr = p*nmo + r
                        qs = q*nmo + s
                        pq = p*nmo + q
                        rs = r*nmo + s
                        if pr>=qs and rs > pq:   
                            t2x = packed_amps[idx]
                            if pr>qs and rs > pq:
                                orb_qA = 2*q
                                orb_pA = 2*p
                                orb_rA = 2*r
                                orb_sA = 2*s
                                t2AA = FermionOperator(((orb_rA,1),(orb_sA,1),(orb_qA,0),(orb_pA,0)),t2x)
    
                                orb_pA = 2*p
                                orb_qB = 2*q+1
                                orb_rA = 2*r
                                orb_sB = 2*s+1
                                t2AB = FermionOperator(((orb_rA,1),(orb_sB,1),(orb_qB,0),(orb_pA,0)),t2x)
                                
                                orb_pB = 2*p+1
                                orb_qA = 2*q
                                orb_rB = 2*r+1
                                orb_sA = 2*s
                                t2BA = FermionOperator(((orb_sA,1),(orb_rB,1),(orb_pB,0),(orb_qA,0)),t2x)
                                                  
                                orb_pB = 2*p+1
                                orb_qB = 2*q+1
                                orb_rB = 2*r+1
                                orb_sB = 2*s+1
                                t2BB = FermionOperator(((orb_rB,1),(orb_sB,1),(orb_qB,0),(orb_pB,0)),t2x)
                                
                                du += t2AA + t2AB + t2BA + t2BB
                            else:
                                orb_pA = 2*p
                                orb_qB = 2*q+1
                                orb_rA = 2*r
                                orb_sB = 2*s+1
                                t2AB = FermionOperator(((orb_rA,1),(orb_sB,1),(orb_qB,0),(orb_pA,0)),t2x)
                                du += t2AB
                            idx += 1
        hat_T = su + du
        assert(idx==len(packed_amps))
        return hat_T