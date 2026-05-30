"""
QAOA Benchmark on Specific Paper Topologies
===========================================
Runs the full MWCSP QAOA formulation on the exactly specified
topologies from the paper to observe real-world heuristic degradation.
"""

import numpy as np
import itertools
import time
from collections import defaultdict

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

# ─────────────────────────────────────────────
#  Paper Topologies Definition
# ─────────────────────────────────────────────

def get_paper_topologies():
    return {
        "Weighted Triangle": {
            "N": 3,
            "edges": [(0, 1, 1), (1, 2, 2), (2, 0, 3)]
        },
        "Heavy Star S4": {
            "N": 4,
            "edges": [(0, 1, 4), (0, 2, 2), (0, 3, 5)]
        },
        "Diagonal Square": {
            "N": 4,
            "edges": [(0, 1, 2), (1, 2, 3), (2, 3, 2), (3, 0, 3), (0, 2, 10)]
        },
        "Path P5": {
            "N": 5,
            "edges": [(0, 1, 3), (1, 2, 7), (2, 3, 2), (3, 4, 6)]
        },
        "Frustrated K4": {
            "N": 4,
            # Complete graph with random uniform weights
            "edges": [(0, 1, 4), (0, 2, 5), (0, 3, 3), (1, 2, 6), (1, 3, 2), (2, 3, 4)]
        }
    }


def brute_force_mwcsp(n, K, edges):
    best_score = -1
    best_col   = None
    for coloring in itertools.product(range(K), repeat=n):
        score = sum(w for (i, j, w) in edges if coloring[i] != coloring[j])
        if score > best_score:
            best_score = score
            best_col   = coloring
    return best_score, best_col

# ─────────────────────────────────────────────
#  QUBO / Ising Construction
# ─────────────────────────────────────────────

def qubit_idx(node, color, K):
    return node * K + color

def build_ising(n, K, edges, lam):
    num_q = n * K
    Q = np.zeros((num_q, num_q))

    # Penalty
    for i in range(n):
        for ka in range(K):
            qa = qubit_idx(i, ka, K)
            Q[qa, qa] += -lam
            for kb in range(ka + 1, K):
                qb = qubit_idx(i, kb, K)
                Q[qa, qb] += 2 * lam

    # Cost
    for (i, j, w) in edges:
        for k in range(K):
            qi = qubit_idx(i, k, K)
            qj = qubit_idx(j, k, K)
            q_lo, q_hi = min(qi, qj), max(qi, qj)
            Q[q_lo, q_hi] += w

    h, J = {}, {}
    for q in range(num_q):
        val = -Q[q, q] / 2.0
        if abs(val) > 1e-9:
            h[q] = h.get(q, 0) + val
        for r in range(q + 1, num_q):
            val_J = Q[q, r] / 4.0
            if abs(val_J) > 1e-9:
                J[(q, r)] = J.get((q, r), 0) + val_J
            val_h_q = -Q[q, r] / 4.0
            val_h_r = -Q[q, r] / 4.0
            if abs(val_h_q) > 1e-9:
                h[q] = h.get(q, 0) + val_h_q
            if abs(val_h_r) > 1e-9:
                h[r] = h.get(r, 0) + val_h_r

    return h, J

def build_qaoa_circuit(num_q, h, J, p):
    gamma = ParameterVector('g', p)
    beta  = ParameterVector('b', p)

    qc = QuantumCircuit(num_q)
    qc.h(range(num_q))

    for layer in range(p):
        for q, coeff in h.items():
            qc.rz(2 * gamma[layer] * coeff, q)
        for (q1, q2), coeff in J.items():
            qc.cx(q1, q2)
            qc.rz(2 * gamma[layer] * coeff, q2)
            qc.cx(q1, q2)
        for q in range(num_q):
            qc.rx(2 * beta[layer], q)

    qc.measure_all()
    return qc, gamma, beta

def run_qaoa(num_q, h, J, p, shots=2048, max_iter=200):
    simulator = AerSimulator(method='statevector')
    qc, gamma, beta = build_qaoa_circuit(num_q, h, J, p)

    def cost_fn(params):
        g_vals = params[:p]
        b_vals = params[p:]
        bound = qc.assign_parameters(
            {gamma[i]: g_vals[i] for i in range(p)} |
            {beta[i]:  b_vals[i] for i in range(p)}
        )
        job    = simulator.run(bound, shots=1024)
        counts = job.result().get_counts()
        total = 0
        total_shots = sum(counts.values())
        for bitstring, cnt in counts.items():
            z = np.array([1 - 2 * int(b) for b in reversed(bitstring)])
            e = sum(coeff * z[q] for q, coeff in h.items())
            e += sum(coeff * z[q1] * z[q2] for (q1, q2), coeff in J.items())
            total += e * cnt
        return total / total_shots

    rng = np.random.default_rng(42)
    x0  = rng.uniform(-np.pi, np.pi, 2 * p)
    res = minimize(cost_fn, x0, method='COBYLA', options={'maxiter': max_iter, 'rhobeg': 0.5})

    best_g = res.x[:p]
    best_b = res.x[p:]
    bound  = qc.assign_parameters(
        {gamma[i]: best_g[i] for i in range(p)} |
        {beta[i]:  best_b[i] for i in range(p)}
    )
    job    = simulator.run(bound, shots=shots)
    return job.result().get_counts()

def decode_one_hot(bitstring, n, K):
    bits = [int(b) for b in reversed(bitstring)]
    coloring = []
    for i in range(n):
        chunk = bits[i * K:(i + 1) * K]
        ones  = [k for k, v in enumerate(chunk) if v == 1]
        if len(ones) != 1:
            return None
        coloring.append(ones[0])
    return tuple(coloring)

def coloring_weight(coloring, edges):
    if coloring is None: return 0
    return sum(w for (i, j, w) in edges if coloring[i] != coloring[j])

# ─────────────────────────────────────────────
#  Run Specific Benchmarks
# ─────────────────────────────────────────────

def run_specific_benchmarks(K=3, p=3, shots=2048):
    topologies = get_paper_topologies()
    
    print("=" * 85)
    print(f" QAOA (p={p}) on PAPER TOPOLOGIES (K={K} Colors, {shots} shots)")
    print("=" * 85)
    header = f"{'Topology Name':<20} | {'Nodes':<5} | {'Edges':<5} | {'Qubits':<6} | {'Opt_W':<6} | " \
             f"{'Valid%':<8} | {'Success%':<10} | {'Time(s)':<8}"
    print(header)
    print("-" * 85)

    lam = 15.0  # Penalty weight

    for name, data in topologies.items():
        N = data["N"]
        edges = data["edges"]
        
        t0 = time.time()
        opt_score, opt_col = brute_force_mwcsp(N, K, edges)

        num_q = N * K
        h_ising, J_ising = build_ising(N, K, edges, lam)

        try:
            counts = run_qaoa(num_q, h_ising, J_ising, p=p, shots=shots)
        except Exception as ex:
            print(f"{name:<20} | ERROR: {ex}")
            continue

        elapsed = time.time() - t0

        total_shots = sum(counts.values())
        success_shots = 0
        valid_shots   = 0

        for bitstring, cnt in counts.items():
            col = decode_one_hot(bitstring, N, K)
            if col is None:
                continue
            valid_shots += cnt
            w = coloring_weight(col, edges)
            if w == opt_score:
                success_shots += cnt

        valid_pct   = 100.0 * valid_shots / total_shots if total_shots else 0
        success_pct = 100.0 * success_shots / total_shots if total_shots else 0

        print(f"{name:<20} | {N:<5} | {len(edges):<5} | {num_q:<6} | {opt_score:<6} | "
              f"{valid_pct:>7.2f}% | {success_pct:>9.2f}% | {elapsed:>8.2f}")

if __name__ == "__main__":
    run_specific_benchmarks(K=3, p=3, shots=4096)
