"""
   Generate base unitary coupled cluster ansatz.
"""
import time
import numpy
from mindquantum.core.operators import TimeEvolution,FermionOperator
from mindquantum.core.operators.utils import hermitian_conjugated
from mindquantum.algorithm.nisq.chem.transform import Transform
from mindquantum import Circuit, X
from QC5.ansatze.hea_and_hva.hf_cir import hf_input
# father class copy RuCCGSD2 as an example
"""
the difference of different ansatz are these functions:
get_amps_num(self),get_packed_amps(self),general_uccsd_generator(self),
need rewrite in different ansatz
"""

def get_random_rccsd_type_int_t1_t2(nelec,norb):
    # some times, ccsd is hard to convergence
    na,nb = nelec
    no = na
    nv = norb - no
    t1_num = no*nv
    t2_num = no*no*nv*nv
    t1_shape = (no,nv)
    t2_shape = (no,no,nv,nv)
    random_t1 = numpy.random.uniform(-1,1,t1_num)*numpy.pi
    random_t2 = numpy.random.uniform(-1,1,t2_num)*numpy.pi
    t1 = random_t1.reshape(t1_shape)
    t2 = random_t2.reshape(t2_shape)
    return t1,t2

def get_random_uccsd_type_int_t1_t2(nelec,norb):
    # some times, ccsd is hard to convergence
    na,nb = nelec
    noA = na
    nvA = norb - noA
    noB = nb
    nvB = norb - noB
    
    t1A_num = noA*nvA
    t1B_num = noB*nvB
    t2AA_num = noA*noA*nvA*nvA
    t2AB_num = noA*noB*nvA*nvB
    t2BB_num = noB*noB*nvB*nvB
    
    random_t1A = numpy.random.uniform(-1,1,t1A_num )*numpy.pi
    random_t1B = numpy.random.uniform(-1,1,t1B_num )*numpy.pi
    random_t2AA = numpy.random.uniform(-1,1,t2AA_num)*numpy.pi
    random_t2AB = numpy.random.uniform(-1,1,t2AB_num)*numpy.pi
    random_t2BB = numpy.random.uniform(-1,1,t2BB_num)*numpy.pi
    
    t1A = random_t1A.reshape((noA,nvA))
    t1B = random_t1B.reshape((noB,nvB))
    t2AA = random_t2AA.reshape((noA,noA,nvA,nvA))
    t2AB = random_t2AB.reshape((noA,noB,nvA,nvB))
    t2BB = random_t2BB.reshape((noB,noB,nvB,nvB))    
    
    t1 = (t1A,t1B)
    t2 = (t2AA,t2AB,t2BB)
    return t1,t2

class UnitaryCCSD:
    
    def __init__(self, nelec, t1, t2, fermion_transform):
        self.t1 = t1
        self.t2 = t2 
        self.nelec = nelec 
        self.fermion_transform = fermion_transform
        
        if type(self.t1) == tuple:
            noa,nva = self.t1[0].shape
            self.nqubits = (noa+nva)*2
        else:
            no,nv = self.t1.shape
            self.nqubits = (no+nv)*2
            
        # for generalized amplitudes, random means each call produce different data
        self.parameter = 1.0e-2 
        
        self.params_circuit = self.get_total_circuit()
        self.params_name = self.params_circuit.params_name
        
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

    def get_ccsd_initial_amps(self):
        initial_amplitudes = self.get_packed_amps()
        # for mindquantum 
        new_order = []
        for temp in self.params_name:
            temp = temp.split('p',) # note the params_name is 'p'
            ordin = eval(temp[1])
            new_order.append(ordin)
        new_amps = numpy.zeros_like(initial_amplitudes)
        for idx,ordin in enumerate(new_order):
            new_amps[idx] = initial_amplitudes[ordin]    
        initial_amplitudes = new_amps    
        return initial_amplitudes    
    
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
        assert(idx==len(packed_amps))
        hat_T = su + du
        return hat_T
    
    def general_uccsd_generator(self,param_expression=True):
        hat_T = self.general_ccsd_generator(param_expression)
        generator = hat_T - hermitian_conjugated(hat_T)
        return generator
    
    def general_qubit_generator(self,param_expression=True):
        """return the qubit_generator under the corresponding fermion_transform
        """
        fermion_generator = self.general_uccsd_generator(param_expression)
        fermion_transform = self.fermion_transform
        if fermion_transform == 'jordan_wigner':
            qubit_generator = Transform(fermion_generator).jordan_wigner()           
        elif fermion_transform == 'parity':
            qubit_generator = Transform(fermion_generator).parity()
        elif fermion_transform == 'bravyi_kitaev':
            qubit_generator = Transform(fermion_generator).bravyi_kitaev()
        elif fermion_transform == 'bravyi_kitaev_tree':
            qubit_generator = Transform(fermion_generator).bravyi_kitaev_tree()
        qubit_generator.compress()
        return qubit_generator  
    
    def get_total_circuit(self,param_expression=True,ref='HF'):
        """
        total_circuit = e^{T-T^\dagger}|ref>=e^{ia_{k}P_{k}}|ref>
        |ref> usually be HF, TimeEvolution simulate e^{-iHt} set t=1, 
        T-T^\dagger = a_k*i*P_k (after transform)
        -iHt = T-T^\dagger, H = i(T-T^\dagger) = i*a_k*i*P_k  = -a_k*P_k
        """
        hf_circuit = hf_input(self.nqubits,self.nelec,ref)
        
        qubit_generator = self.general_qubit_generator(param_expression)
        # Take the imaginary part and multiply -1 since we are simulating exp(-iHt)
        qubit_generator = -1*qubit_generator.imag
        ansatz_circuit_trottered = TimeEvolution(qubit_generator, 1.0).circuit
        total_circuit = hf_circuit + ansatz_circuit_trottered
        return total_circuit
    
    def analyze_psi(self, input_amps=None, coeff_thres=0.1):
        """analyze the wavefunction into occupation number vectors"""
        # get psi from input_amps
        pr = dict(zip(self.params_name,input_amps))
        psi = self.params_circuit.get_qs(pr=pr, ket=False)
        # analyze psi
        n_qubits = self.nqubits
        Nalpha, Nbeta = self.nelec
        # check 
        check=0
        for i in range(2**n_qubits):          
            tmp = "{:0"+str(n_qubits)+"b}"
            xt  = list(tmp.format(i))
            for idx,j in enumerate(xt):
                xt[idx] = eval(j)
            xxt = numpy.array(xt)
            even_sum = numpy.sum(xxt[::2]) 
            odd_sum = numpy.sum(xxt[1::2])
            if even_sum == Nalpha and odd_sum == Nbeta:
                txt = psi[i]
                check += numpy.abs(txt)**2
                if abs(txt)>=coeff_thres:
                    print("ONVs:{} coeff:{: .9f} population:{:.4%}"\
                          .format(str(list(xt)),txt,numpy.abs(txt)**2))
        ok = numpy.isclose(check,1.0)
        if ok == False:
            print('Notice: the sum of population is ',check,'Please check your code (TOT)')
        return None