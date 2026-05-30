import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
import warnings
warnings.filterwarnings("ignore")

def create_maxcut_pauli(N, edges):
    """
    Creates the Max-Cut Ising Hamiltonian as a SparsePauliOp.
    Max-Cut cost is C = sum w_ij (1 - Z_i Z_j) / 2.
    We want to maximize C, which means minimizing H = sum w_ij Z_i Z_j.
    """
    paulis = []
    coeffs = []
    
    # Calculate offset for the true max cut value
    offset = 0
    for (i, j), w in edges.items():
        offset += w / 2
        
        # Create Z_i Z_j string
        pauli_str = ['I'] * N
        pauli_str[i] = 'Z'
        pauli_str[j] = 'Z'
        pauli_str = ''.join(pauli_str)[::-1] # Qiskit endianness
        
        paulis.append(pauli_str)
        coeffs.append(w / 2) # we minimize Z_i Z_j to maximize cut
        
    return SparsePauliOp(paulis, coeffs), offset

def create_qaoa_circuit(N, edges, p, gammas, betas):
    """
    Builds the QAOA QuantumCircuit for Max-Cut.
    """
    qc = QuantumCircuit(N)
    # Initial state |+...+>
    qc.h(range(N))
    
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Cost Hamiltonian (e^{-i gamma w_{ij} Z_i Z_j})
        for (i, j), w in edges.items():
            qc.cx(i, j)
            qc.rz(2 * gamma * w, j)
            qc.cx(i, j)
            
        # Mixer Hamiltonian (e^{-i beta X})
        for i in range(N):
            qc.rx(2 * beta, i)
            
    return qc

def optimize_qaoa_qiskit(N, edges, p):
    operator, offset = create_maxcut_pauli(N, edges)
    estimator = Estimator()
    
    def objective_function(params):
        gammas = params[:p]
        betas = params[p:]
        qc = create_qaoa_circuit(N, edges, p, gammas, betas)
        
        # Estimator computes <psi|H|psi>. We want to minimize this.
        result = estimator.run(qc, operator).result()
        return result.values[0]

    best_val = float('inf')
    
    # Random restarts to avoid local minima
    for _ in range(5):
        initial_params = np.random.uniform(0, np.pi, 2*p)
        res = minimize(objective_function, initial_params, method='COBYLA')
        if res.fun < best_val:
            best_val = res.fun
            
    # Convert expectation of sum w_ij Z_i Z_j back to Max-Cut score
    max_cut_expectation = offset - best_val
    
    # Calculate exact Max-Cut classically for alpha
    dim = 2**N
    max_cut_exact = 0
    for state in range(dim):
        cost = 0
        for (i, j), w in edges.items():
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i != bit_j:
                cost += w
        if cost > max_cut_exact:
            max_cut_exact = cost
            
    alpha = max_cut_expectation / max_cut_exact if max_cut_exact > 0 else 0
    return max_cut_expectation, max_cut_exact, alpha

graphs = {
    "Weighted Triangle (N=3)": (3, {(0,1):1, (1,2):2, (0,2):3}),
    "Heavy Star S4 (N=4)": (4, {(0,1):4, (0,2):2, (0,3):5}),
    "Diagonal Square (N=4)": (4, {(0,1):2, (1,2):2, (2,3):2, (3,0):2, (0,2):10}),
    "Path P5 (N=5)": (5, {(0,1):3, (1,2):7, (2,3):2, (3,4):6}),
    "Frustrated K4 (N=4)": (4, {(0,1):2, (0,2):10, (0,3):10, (1,2):10, (1,3):10, (2,3):10})
}

for name, (N, edges) in graphs.items():
    print(f"--- {name} ---")
    val1, max_val, alpha1 = optimize_qaoa_qiskit(N, edges, 1)
    print(f"p=1: Max={max_val}, alpha={alpha1:.4f}")
    val3, _, alpha3 = optimize_qaoa_qiskit(N, edges, 3)
    print(f"p=3: alpha={alpha3:.4f}")
    print()
