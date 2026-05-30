"""
Exact Transpiled Gate Count for QAOA (p=7) — Standard Hardware Basis
Uses generate_preset_pass_manager with optimization_level=3
and basis_gates=['cx', 'id', 'rz', 'sx', 'x'] (IBM standard).
"""
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.transpiler import generate_preset_pass_manager

# ── Topologies ──────────────────────────────────────────────────
def get_paper_topologies():
    return {
        "Weighted Triangle": {"N": 3, "edges": [(0, 1, 1), (1, 2, 2), (2, 0, 3)]},
        "Heavy Star S4":     {"N": 4, "edges": [(0, 1, 4), (0, 2, 2), (0, 3, 5)]},
        "Diagonal Square":   {"N": 4, "edges": [(0, 1, 2), (1, 2, 3), (2, 3, 2), (3, 0, 3), (0, 2, 10)]},
        "Frustrated K4":     {"N": 4, "edges": [(0, 1, 4), (0, 2, 5), (0, 3, 3), (1, 2, 6), (1, 3, 2), (2, 3, 4)]},
        "Path P5":           {"N": 5, "edges": [(0, 1, 3), (1, 2, 7), (2, 3, 2), (3, 4, 6)]},
    }

# ── Ising Builder ───────────────────────────────────────────────
def qubit_idx(node, color, K):
    return node * K + color

def build_ising(n, K, edges, lam=15.0):
    num_q = n * K
    Q = np.zeros((num_q, num_q))
    for i in range(n):
        for ka in range(K):
            qa = qubit_idx(i, ka, K)
            Q[qa, qa] += -lam
            for kb in range(ka + 1, K):
                qb = qubit_idx(i, kb, K)
                Q[qa, qb] += 2 * lam
    for (i, j, w) in edges:
        for k in range(K):
            qi = min(qubit_idx(i, k, K), qubit_idx(j, k, K))
            qj = max(qubit_idx(i, k, K), qubit_idx(j, k, K))
            Q[qi, qj] += w
    h, J = {}, {}
    for q in range(num_q):
        val = -Q[q, q] / 2.0
        if abs(val) > 1e-9:
            h[q] = h.get(q, 0) + val
        for r in range(q + 1, num_q):
            if abs(Q[q, r] / 4.0) > 1e-9:
                J[(q, r)] = Q[q, r] / 4.0
            val_h = -Q[q, r] / 4.0
            if abs(val_h) > 1e-9:
                h[q] = h.get(q, 0) + val_h
                h[r] = h.get(r, 0) + val_h
    return h, J

# ── Preset Pass Manager (same as friend's code) ────────────────
PM = generate_preset_pass_manager(
    optimization_level=3,
    basis_gates=['cx', 'id', 'rz', 'sx', 'x']
)

def transpiled_metrics(qc):
    qc_t = PM.run(qc)
    ops = qc_t.count_ops()
    return {
        'depth': qc_t.depth(),
        'cx': ops.get('cx', 0),
        'gates': qc_t.size(),
        'ops': dict(ops),
    }

# ── Main ────────────────────────────────────────────────────────
def run_all():
    topologies = get_paper_topologies()
    K = 3
    p = 7

    print("=" * 100)
    print(f" QAOA (p={p}) — EXACT TRANSPILED GATE COUNT")
    print(f" Basis: [cx, id, rz, sx, x]  |  optimization_level=3")
    print("=" * 100)

    header = (f"{'Topology':<20} | {'Qubits':<7} | {'Raw Gates':<10} | "
              f"{'Trans. Gates':<13} | {'Trans. Depth':<13} | {'CX count':<10} | Gate Breakdown")
    print(header)
    print("-" * 100)

    np.random.seed(42)

    for name, data in topologies.items():
        N = data["N"]
        edges = data["edges"]
        num_q = N * K

        h, J = build_ising(N, K, edges)

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

        # Bind random angles (structure doesn't change with angles)
        angles = np.random.uniform(-np.pi, np.pi, 2 * p)
        bound_qc = qc.assign_parameters(angles)

        m = transpiled_metrics(bound_qc)

        print(f"{name:<20} | {num_q:<7} | {raw_gates:<10} | "
              f"{m['gates']:<13} | {m['depth']:<13} | {m['cx']:<10} | {m['ops']}")

    print("=" * 100)

if __name__ == "__main__":
    run_all()
