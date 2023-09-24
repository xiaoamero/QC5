"""Spin Unrestricted unitary Coupled Cluster Generalized Singles ansatz."""

import numpy
from mindquantum.core.operators import TimeEvolution,FermionOperator
from mindquantum.core.operators.utils import hermitian_conjugated
from mindquantum.algorithm.nisq.chem.transform import Transform
from mindquantum import Circuit, X
from .UnitaryCCSD_base import UnitaryCCSD

class UuCCGSD1complex(UnitaryCCSD):

    def __init__(self, nelec, t1, t2, fermion_transform):
        super().__init__(nelec, t1, t2, fermion_transform)
        
    def get_amps_num(self):
        noA,noB,nvA,nvB = self.t2[1].shape
        nmo = noA + nvA
        t1_num = nmo*(nmo-1)//2 + nmo*(nmo-1)//2
        t2_num = (noA*(noA-1)//2)*(nvA*(nvA-1)//2) + noA*noB*nvA*nvB + (noB*(noB-1)//2)*(nvB*(nvB-1)//2)
        num = t1_num + t2_num
        complex_num = num*2
        return complex_num
    
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
        # t1a_list
        for pA in range(nmo):        
            for qA in range(pA):
                if pA >= noA and qA < noA:
                    usa = t1a[qA,pA-noA]
                else:
                    usa = numpy.random.uniform(-1,1)*self.parameter
                t1a_list.append(usa)

        # t1b_list
        for pB in range(nmo):        
            for qB in range(pB):
                if pB >= noB and qB < noB:
                    usb = t1b[qB,pB-noB]
                else:
                    usb = numpy.random.uniform(-1,1)*self.parameter
                t1b_list.append(usb)
        
        #t2aa_list       
        for iA in range(noA):
            for jA in range(iA):
                for aA in range(nvA):
                    for bA in range(aA):
                        usaa = t2aa[iA,jA,aA,bA] 
                        t2aa_list.append(usaa)
                                        
        # t2ab_list
        for iA in range(noA):
            for jB in range(noB):
                for aA in range(nvA):
                    for bB in range(nvB):
                        usab = t2ab[iA,jB,aA,bB]
                        t2ab_list.append(usab)
        
        # t2bb_list        
        for iB in range(noB):
            for jB in range(iB):
                for aB in range(nvB):
                    for bB in range(aB):
                        usbb = t2bb[iB,jB,aB,bB] 
                        t2bb_list.append(usbb)
    
        packed_amps = (t1a_list+t1b_list+t2aa_list+t2ab_list+t2bb_list)*2 #complex,imag = real
        packed_amps = numpy.array(packed_amps)
        packed_amps[abs(packed_amps) < 1e-8] = 0
        return packed_amps
    
    def general_ccsd_generator(self,param_expression=True):
        """ return the fermion_generator of corresponding ansatz,
            packed_amps = ["p1", "p2", "p3", ...] """
        noA,noB,nvA,nvB = self.t2[1].shape
        amps_num = self.get_amps_num()
        med = amps_num//2
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
                t1A = FermionOperator(((orb_pA,1),(orb_qA,0)),packed_amps[idx])
                t1A += FermionOperator(((orb_pA,1),(orb_qA,0)),packed_amps[idx+med])*1j
                usa += t1A
                idx += 1
        # t1b
        usb = FermionOperator()
        for pB in range(nmo):
            for qB in range(pB):
                orb_pB = 2*pB+1
                orb_qB = 2*qB+1
                t1B = FermionOperator(((orb_pB,1),(orb_qB,0)),packed_amps[idx])
                t1B += FermionOperator(((orb_pB,1),(orb_qB,0)),packed_amps[idx+med])*1j
                usb += t1B
                idx += 1  
        # t2aa
        usaa = FermionOperator()
        for iA in range(noA):
            for jA in range(iA):
                for aA in range(nvA):
                    for bA in range(aA):
                        orb_iA = 2*(iA)
                        orb_jA = 2*(jA)
                        orb_aA = 2*(aA+noA)
                        orb_bA = 2*(bA+noA)
                        t2AA = FermionOperator(((orb_aA,1),(orb_bA,1),(orb_jA,0),(orb_iA,0)),packed_amps[idx])
                        t2AA += FermionOperator(((orb_aA,1),(orb_bA,1),(orb_jA,0),(orb_iA,0)),packed_amps[idx+med])*1j
                        usaa += t2AA
                        idx += 1 
        #t2ab
        usab = FermionOperator()
        for iA in range(noA):
            for jB in range(noB):
                for aA in range(nvA):
                    for bB in range(nvB):
                        orb_iA = 2*(iA)
                        orb_jB = 2*(jB)+1
                        orb_aA = 2*(aA+noA)
                        orb_bB = 2*(bB+noB)+1
                        t2AB = FermionOperator(((orb_aA,1),(orb_bB,1),(orb_jB,0),(orb_iA,0)),packed_amps[idx])
                        t2AB += FermionOperator(((orb_aA,1),(orb_bB,1),(orb_jB,0),(orb_iA,0)),packed_amps[idx+med])*1j
                        usab += t2AB
                        idx += 1     
        # t2bb
        usbb = FermionOperator()
        for iB in range(noB):
            for jB in range(iB):
                for aB in range(nvB):
                    for bB in range(aB):
                        orb_iB = 2*(iB)+1
                        orb_jB = 2*(jB)+1
                        orb_aB = 2*(aB+noB)+1
                        orb_bB = 2*(bB+noB)+1  
                        t2BB = FermionOperator(((orb_aB,1),(orb_bB,1),(orb_jB,0),(orb_iB,0)),packed_amps[idx])
                        t2BB += FermionOperator(((orb_aB,1),(orb_bB,1),(orb_jB,0),(orb_iB,0)),packed_amps[idx+med])*1j
                        usbb += t2BB
                        idx += 1          
        assert(2*idx==len(packed_amps))
        hat_T = usa + usb + usaa + usab + usbb
        return hat_T