"""
   Generate spin un-restricted unitary coupled cluster singles and doubles ansatz.
"""
import numpy
from mindquantum.core.operators import TimeEvolution,FermionOperator
from mindquantum.core.operators.utils import hermitian_conjugated
from mindquantum.algorithm.nisq.chem.transform import Transform
from mindquantum import Circuit, X
from .UnitaryCCSD_base import UnitaryCCSD

class UuCCSD(UnitaryCCSD):
    
    def __init__(self, nelec, t1, t2, fermion_transform):
        super().__init__(nelec, t1, t2, fermion_transform)

    def get_amps_num(self):
        noA,noB,nvA,nvB = self.t2[1].shape
        novA = noA*nvA
        novB = noB*nvB
        t1_num = novA + novB
        t2_num = (noA*(noA-1)//2)*(nvA*(nvA-1)//2) + noA*noB*nvA*nvB + (noB*(noB-1)//2)*(nvB*(nvB-1)//2)
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
        # t1a_list
        for iA in range(noA):        
            for aA in range(nvA):
                usa = t1a[iA,aA]
                t1a_list.append(usa)
        
        # t1b_list
        for iB in range(noB):        
            for aB in range(nvB):
                usb = t1b[iB,aB]
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
        idx = 0
        # t1a
        usa = FermionOperator() 
        for iA in range(noA):
            for aA in range(nvA):
                orb_iA = 2*iA
                orb_aA = 2*(aA+noA)
                usa += FermionOperator(((orb_aA,1),(orb_iA,0)),packed_amps[idx])
                idx += 1
        # t1b
        usb = FermionOperator()
        for iB in range(noB):
            for aB in range(nvB):
                orb_iB = 2*iB+1
                orb_aB = 2*(aB+noB)+1   
                usb += FermionOperator(((orb_aB,1),(orb_iB,0)),packed_amps[idx])
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
                        usaa += FermionOperator(((orb_aA,1),(orb_bA,1),(orb_jA,0),(orb_iA,0)),packed_amps[idx])
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
                        usab += FermionOperator(((orb_aA,1),(orb_bB,1),(orb_jB,0),(orb_iA,0)),packed_amps[idx])
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
                        usbb += FermionOperator(((orb_aB,1),(orb_bB,1),(orb_jB,0),(orb_iB,0)),packed_amps[idx])
                        idx += 1          
        hat_T = usa + usb + usaa + usab + usbb         
        assert(idx==len(packed_amps))
        return hat_T