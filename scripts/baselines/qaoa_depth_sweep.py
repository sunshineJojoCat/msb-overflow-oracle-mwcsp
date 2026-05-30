"""
QAOA Depth Sweep (p=1..7) — Transpiled Gate Counts + Statevector Success Rate
==============================================================================
Shows how increasing p affects both circuit cost and solution quality.
Uses statevector simulation for exact probabilities (no shot noise).
Transpiles to standard hardware basis [cx, rz, sx, x] at optimization_level=3.
"""
import sys, time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.transpiler import generate_preset_pass_manager

# ── Add local path for helper imports ───────────────────────────
sys.path.insert(0, sys.path[0])  # ensure local dir is on path
from simulate_qaoa_paper_topologies import build_ising, brute_force_mwcsp

# ── Topologies ──────────────────────────────────────────────────
def get_paper_topologies():
    return {
        "Weighted Triangle": {"N": 3, "edges": [(0, 1, 1), (1, 2, 2), (2, 0, 3)]},
        "Heavy Star S4":     {"N": 4, "edges": [(0, 1, 4), (0, 2, 2), (0, 3, 5)]},
        "Diagonal Square":   {"N": 4, "edges": [(0, 1, 2), (1, 2, 3), (2, 3, 2), (3, 0, 3), (0, 2, 10)]},
        "Frustrated K4":     {"N": 4, "edges": [(0, 1, 4), (0, 2, 5), (0, 3, 3), (1, 2, 6), (1, 3, 2), (2, 3, 4)]},
        "Path P5":           {"N": 5, "edges": [(0, 1, 3), (1, 2, 7), (2, 3, 2), (3, 4, 6)]},
    }

# ── Transpiler ──────────────────────────────────────────────────
PM = generate_preset_pass_manager(
    optimization_level=3,
    basis_gates=['cx', 'id', 'rz', 'sx', 'x']
)

def decode_one_hot(idx, n, K):
    bitstring = format(idx, f'0{n*K}b')[::-1]
    bits = [int(b) for b in bitstring]
    coloring = []
    for i in range(n):
        chunk = bits[i * K:(i + 1) * K]
        ones = [k for k, v in enumerate(chunk) if v == 1]
        if len(ones) != 1:
            return None
        coloring.append(ones[0])
    return tuple(coloring)

# ── Main Sweep ──────────────────────────────────────────────────
def run_sweep():
    topologies = get_paper_topologies()
    K = 3
    lam = 15.0
    simulator = AerSimulator(method='statevector')
    p_values = list(range(1, 8))  # p = 1 to 7

    print("=" * 110)
    print(" QAOA DEPTH SWEEP (p=1..7) — EXACT STATEVECTOR + TRANSPILED GATE COUNTS")
    print(" Basis: [cx, rz, sx, x]  |  optimization_level=3  |  5 random seeds per (topology, p)")
    print("=" * 110)

    for name, data in topologies.items():
        N = data["N"]
        edges = data["edges"]
        num_q = N * K
        opt_w, _ = brute_force_mwcsp(N, K, edges)

        h, J = build_ising(N, K, edges, lam)

        # Pre-compute energy landscape (once per topology)
        energies = np.zeros(2**num_q)
        for i in range(2**num_q):
            bs = format(i, f'0{num_q}b')[::-1]
            z = np.array([1 - 2 * int(b) for b in bs])
            energies[i] = (sum(c * z[q] for q, c in h.items())
                         + sum(c * z[q1] * z[q2] for (q1, q2), c in J.items()))

        print(f"\n{'─'*110}")
        print(f"  {name}  (N={N}, |E|={len(edges)}, Qubits={num_q}, Optimal Weight={opt_w})")
        print(f"{'─'*110}")
        print(f"  {'p':<4} | {'Raw Gates':<10} | {'Trans. Gates':<13} | {'Trans. Depth':<13} | "
              f"{'CX count':<9} | {'Invalid%':<10} | {'Success%':<10} | {'Time(s)':<8}")
        print(f"  {'─'*104}")

        for p in p_values:
            t0 = time.time()

            # Build parameterized circuit
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

            raw_gates = sum(qc.count_ops().values())

            # Optimize with statevector (multi-seed)
            qc_sv = qc.copy()
            qc_sv.save_statevector()

            def cost_fn(params):
                bound = qc_sv.assign_parameters(params)
                sv = simulator.run(bound).result().get_statevector()
                probs = np.abs(sv)**2
                return np.sum(probs * energies)

            np.random.seed(99)
            best_res = None
            for seed in range(5):
                x0 = np.random.uniform(-np.pi, np.pi, 2 * p)
                res = minimize(cost_fn, x0, method='L-BFGS-B',
                             options={'maxiter': 300})
                if best_res is None or res.fun < best_res.fun:
                    best_res = res

            # Evaluate final probabilities
            bound_final = qc_sv.assign_parameters(best_res.x)
            probs = np.abs(simulator.run(bound_final).result().get_statevector())**2

            invalid_prob = 0.0
            success_prob = 0.0
            for i, pv in enumerate(probs):
                col = decode_one_hot(i, N, K)
                if col is None:
                    invalid_prob += pv
                else:
                    w = sum(ww for (u, v, ww) in edges if col[u] != col[v])
                    if w == opt_w:
                        success_prob += pv

            # Transpile (use the non-statevector circuit with bound params)
            bound_qc = qc.assign_parameters(best_res.x)
            qc_t = PM.run(bound_qc)
            ops = qc_t.count_ops()
            trans_gates = qc_t.size()
            trans_depth = qc_t.depth()
            cx_count = ops.get('cx', 0)

            elapsed = time.time() - t0

            print(f"  {p:<4} | {raw_gates:<10} | {trans_gates:<13} | {trans_depth:<13} | "
                  f"{cx_count:<9} | {invalid_prob*100:>8.2f}% | {success_prob*100:>8.2f}% | "
                  f"{elapsed:>7.1f}s")

    print(f"\n{'='*110}")
    print(" Done! Notice how increasing p barely improves Success% while gate costs grow linearly.")
    print(f"{'='*110}")

if __name__ == "__main__":
    run_sweep()
