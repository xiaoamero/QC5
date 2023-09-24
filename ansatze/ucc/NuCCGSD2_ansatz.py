"""
   None of Spin, unitary coupled cluster generalized singles and doubles ansatz.
"""
import numpy
from mindquantum.core.operators import TimeEvolution,FermionOperator
from mindquantum.core.operators.utils import hermitian_conjugated
from mindquantum.algorithm.nisq.chem.transform import Transform
from mindquantum import Circuit, X
from QC5.ansatze.hea_and_hva.hf_cir import hf_input

class NuCCGSD2:
    
    def __init__(self, nqubits, nelec, fermion_transform):
        self.nqubits = nqubits
        self.nelec = nelec
        self.fermion_transform = fermion_transform
        
        self.params_circuit = self.get_total_circuit()
        self.params_name = self.params_circuit.params_name   
    
    def get_amps_num(self):
        num1 = self.nqubits*(self.nqubits-1)//2
        num2 = num1*(num1-1)//2
        num = num1 + num2
        return num
    
    def general_ccsd_generator(self):

        # t1
        idx = 0
        su = FermionOperator() 
        for p in range(self.nqubits):
            for q in range(p):
                t1 = FermionOperator(((p,1),(q,0)),'p'+str(idx))
                su += t1
                idx += 1     
        # t2
        du = FermionOperator() 
        for p in range(self.nqubits):
            for q in range(p):
                for r in range(self.nqubits):
                    for s in range(r):
                        pq = p*(p-1)//2 + q
                        rs = r*(r-1)//2 + s
                        if rs > pq:   
                            t2 = FermionOperator(((r,1),(s,1),(q,0),(p,0)),'p'+str(idx))
                            du += t2
                            idx += 1
        assert(idx==self.get_amps_num())
        hat_T = su + du
        return hat_T
    
    def general_uccsd_generator(self):
        hat_T = self.general_ccsd_generator()
        generator = hat_T - hermitian_conjugated(hat_T)
        return generator
    
    def general_qubit_generator(self):
        """return the qubit_generator under the corresponding fermion_transform
        """
        fermion_generator = self.general_uccsd_generator()
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
    
    def get_total_circuit(self,ref='HF'):
        """
        total_circuit = e^{T-T^\dagger}|ref>=e^{ia_{k}P_{k}}|ref>
        |ref> usually be HF, TimeEvolution simulate e^{-iHt} set t=1, 
        T-T^\dagger = a_k*i*P_k (after transform)
        -iHt = T-T^\dagger, H = i(T-T^\dagger) = i*a_k*i*P_k  = -a_k*P_k
        """
        hf_circuit = hf_input(self.nqubits,self.nelec,ref)
        
        qubit_generator = self.general_qubit_generator()
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