"""Spin Unrestricted unitary Coupled Cluster Generalized Singles and Doubles ansatz."""

import numpy
from mindquantum.core.operators import TimeEvolution,FermionOperator
from mindquantum.core.operators.utils import hermitian_conjugated
from mindquantum.algorithm.nisq.chem.transform import Transform
from mindquantum import Circuit, X
from .UnitaryCCSD_base import UnitaryCCSD

class UuCCGSD2(UnitaryCCSD):

    def __init__(self, nelec, t1, t2, fermion_transform):
        super().__init__(nelec, t1, t2, fermion_transform)
        
    def get_amps_num(self):
        noA,noB,nvA,nvB = self.t2[1].shape
        nmo = noA + nvA
        k1 = nmo*(nmo-1)//2
        k2 = k1*(k1-1)//2
        k3 = nmo*nmo
        k4 = k3*(k3-1)//2
        t1_num = k1 + k1
        t2_num = k2+ k4 + k2
        num = t1_num + t2_num
        return num
    
    def get_packed_amps(self):
        t1a,t1b = self.t1  
        t2aa,t2ab,t2bb= self.t2
        noA,noB,nvA,nvB = self.t2[1].shape
        
        t1a_list = []
        t1b_list = []
        t2aa_list = []
        t2ab_list = []
        t2bb_list = []
        nmo = noA + nvA
        # t1a_list AA, t[p,q]a^p_q
        for pA in range(nmo):        
            for qA in range(pA):
                if pA >= noA and qA < noA:
                    usa = t1a[qA,pA-noA]
                else:
                    usa = numpy.random.uniform(-1,1)*self.parameter
                t1a_list.append(usa)
        # t1b_list BB, t[p',q']a^p'_q'
        for pB in range(nmo):        
            for qB in range(pB):
                if pB >= noB and qB < noB:
                    usb = t1b[qB,pB-noB]
                else:
                    usb = numpy.random.uniform(-1,1)*self.parameter
                t1b_list.append(usb)
        #t2aa_list  
        for r in range(nmo):
            for s in range(r):
                rs = r*(r-1)//2 + s
                for p in range(r+1):
                    for q in range(p):
                        pq = p*(p-1)//2 + q
                        if rs > pq:    
                            if r>=noA and s>=noA and p< noA and q<noA:
                                usaa = t2aa[p,q,r-noA,s-noA] 
                            else:
                                usaa = numpy.random.uniform(-1,1)*self.parameter
                            t2aa_list.append(usaa)     
        # t2ab_list
        for r in range(nmo):
            for s in range(nmo):
                r_s = r*nmo + s
                for p in range(nmo):
                    for q in range(nmo):
                        p_q = p*nmo + q
                        if r_s > p_q:
                            if r>=noA and s>=noB and p< noA and q<noB:
                                usab = t2ab[p,q,r-noA,s-noB] 
                            else:
                                usab = numpy.random.uniform(-1,1)*self.parameter
                            t2ab_list.append(usab)
        # t2bb_list        
        for r in range(nmo):
            for s in range(r):
                rs = r*(r-1)//2 + s
                for p in range(r+1):
                    for q in range(p):
                        pq = p*(p-1)//2 + q
                        if rs > pq:    
                            if r>=noB and s>=noB and p< noB and q<noB:
                                usbb = t2bb[p,q,r-noB,s-noB] 
                            else:
                                usbb = numpy.random.uniform(-1,1)*self.parameter
                            t2bb_list.append(usbb)
    
        packed_amps = t1a_list+t1b_list+t2aa_list+t2ab_list+t2bb_list
        packed_amps = numpy.array(packed_amps)
        packed_amps[abs(packed_amps) < 1e-8] = 0
        return packed_amps
    
    def general_ccsd_generator(self,param_expression=True):
        """ return the fermion_generator of corresponding ansatz,
            packed_amps = ["p1", "p2", "p3", ...] """
        noA,noB,nvA,nvB = self.t2[1].shape
        amps_num = self.get_amps_num()
        if param_expression:
            packed_amps = ["p" + str(i) for i in range(amps_num)]
        else:
            packed_amps = self.get_packed_amps()
        nmo = noA + nvA
        idx = 0
        # t1a
        usa = FermionOperator() 
        for pA in range(nmo):
            for qA in range(pA):
                orb_pA = 2*pA
                orb_qA = 2*qA
                usa += FermionOperator(((orb_pA,1),(orb_qA,0)),packed_amps[idx])
                idx += 1
        # t1b
        usb = FermionOperator()
        for pB in range(nmo):
            for qB in range(pB):
                orb_pB = 2*pB+1
                orb_qB = 2*qB+1
                usb += FermionOperator(((orb_pB,1),(orb_qB,0)),packed_amps[idx])
                idx += 1
        # t2aa
        usaa = FermionOperator()
        for r in range(nmo):
            for s in range(r):
                rs = r*(r-1)//2 + s
                for p in range(r+1):
                    for q in range(p):
                        pq = p*(p-1)//2 + q
                        if rs > pq: 
                            orb_pA = 2*p
                            orb_qA = 2*q
                            orb_rA = 2*r
                            orb_sA = 2*s
                            usaa += FermionOperator(((orb_rA,1),(orb_sA,1),(orb_qA,0),(orb_pA,0)),packed_amps[idx])
                            idx += 1 
        #t2ab
        usab = FermionOperator()
        for r in range(nmo):
            for s in range(nmo):
                r_s = r*nmo + s
                for p in range(nmo):
                    for q in range(nmo):
                        p_q = p*nmo + q
                        if r_s > p_q:
                            orb_pA = 2*p
                            orb_qB = 2*q+1
                            orb_rA = 2*r
                            orb_sB = 2*s+1
                            usab += FermionOperator(((orb_rA,1),(orb_sB,1),(orb_qB,0),(orb_pA,0)),packed_amps[idx])
                            idx += 1
        # t2bb
        usbb = FermionOperator()
        for r in range(nmo):
            for s in range(r):
                rs = r*(r-1)//2 + s
                for p in range(r+1):
                    for q in range(p):
                        pq = p*(p-1)//2 + q
                        if rs > pq: 
                            orb_pB = 2*p+1
                            orb_qB = 2*q+1
                            orb_rB = 2*r+1
                            orb_sB = 2*s+1
                            usbb += FermionOperator(((orb_rB,1),(orb_sB,1),(orb_qB,0),(orb_pB,0)),packed_amps[idx])
                            idx += 1
        assert(idx==len(packed_amps))
        hat_T = usa + usb + usaa + usab + usbb
        return hat_T