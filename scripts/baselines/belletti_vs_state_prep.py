"""
belletti_vs_state_prep.py - Head-to-head on the UNWEIGHTED graph K-coloring
decision problem:

  (A) Belletti original: Hadamard init + per-node invalid-colour checker
      + per-edge collision detector + global MCX over |V|+|E| flags.

  (B) Our full MSB-Overflow Min Width oracle, configured for the unweighted
      decision problem (every edge weight = 1, target = |E|, so MSB triggers
      only when every edge has been properly coloured). This runs our COMPLETE
      pipeline -- Shukla-Vedula state prep, recycled-ancilla edge comparison,
      ripple-carry weight accumulator, single-CNOT MSB phase flip, and
      A_init S_0 A_init^dagger diffusion -- on the exact same problem instance
      Belletti is solving.

Both oracles solve "find a proper K-coloring". Frustrated K4 has none, so both
return zero amplitude there. Reports qubits, transpiled depth (basis
cx/id/rz/sx/x, opt_level=3), and the empirical proper-colouring rate from 2048
noise-free shots at the empirically-optimal Grover R.
"""
import math
import os
import sys
from itertools import product
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# Import our canonical MSB-Overflow oracle constructor from the reproducibility
# package so this script never diverges from the one that produced Tables II,
# VI, VII.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "reproducibility"))
from verify_table2 import create_min_width_oracle  # noqa: E402

BASIS = ["cx", "id", "rz", "sx", "x"]
K = 3
SHOTS = 2048


# ============ Belletti original oracle (vanilla) ============================
def per_node_valid(qc, coloring, node_anc, node, q):
    qs = [coloring[node * q + j] for j in range(q)]
    qc.x(node_anc[node])
    qc.mcx(qs, node_anc[node])


def per_node_valid_inv(qc, coloring, node_anc, node, q):
    qs = [coloring[node * q + j] for j in range(q)]
    qc.mcx(qs, node_anc[node])
    qc.x(node_anc[node])


def per_edge_check(qc, coloring, edge_anc, e_idx, u, v, q):
    u_qs = [coloring[u * q + j] for j in range(q)]
    v_qs = [coloring[v * q + j] for j in range(q)]
    for j in range(q):
        qc.cx(u_qs[j], v_qs[j])
        qc.x(v_qs[j])
    qc.mcx(v_qs, edge_anc[e_idx])
    for j in reversed(range(q)):
        qc.x(v_qs[j])
        qc.cx(u_qs[j], v_qs[j])
    qc.x(edge_anc[e_idx])


def per_edge_check_inv(qc, coloring, edge_anc, e_idx, u, v, q):
    u_qs = [coloring[u * q + j] for j in range(q)]
    v_qs = [coloring[v * q + j] for j in range(q)]
    qc.x(edge_anc[e_idx])
    for j in range(q):
        qc.cx(u_qs[j], v_qs[j])
        qc.x(v_qs[j])
    qc.mcx(v_qs, edge_anc[e_idx])
    for j in reversed(range(q)):
        qc.x(v_qs[j])
        qc.cx(u_qs[j], v_qs[j])


def belletti_original(N, edges, grover_iters):
    """Faithful Belletti reimplementation: Hadamard + node checkers + per-edge
    flags + global MCX phase trigger over |V|+|E| ancillas."""
    q = 2  # K=3
    Nq = N * q
    E = len(edges)
    coloring = QuantumRegister(Nq, "color")
    node_anc = QuantumRegister(N, "nodev")
    edge_anc = QuantumRegister(E, "edgev")
    phase_anc = QuantumRegister(1, "phase")
    cr = ClassicalRegister(Nq, "c")
    qc = QuantumCircuit(coloring, node_anc, edge_anc, phase_anc, cr)

    qc.h(coloring)
    qc.x(phase_anc[0]); qc.h(phase_anc[0])

    for _ in range(grover_iters):
        for v in range(N):
            per_node_valid(qc, coloring, node_anc, v, q)
        for e_idx, (u, vv) in enumerate(edges):
            per_edge_check(qc, coloring, edge_anc, e_idx, u, vv, q)
        all_flags = list(node_anc) + list(edge_anc)
        qc.mcx(all_flags, phase_anc[0])
        for e_idx, (u, vv) in reversed(list(enumerate(edges))):
            per_edge_check_inv(qc, coloring, edge_anc, e_idx, u, vv, q)
        for v in range(N):
            per_node_valid_inv(qc, coloring, node_anc, v, q)
        # Diffusion H S_0 H
        qc.h(coloring)
        qc.x(coloring)
        qc.h(coloring[-1])
        qc.mcx(coloring[:-1], coloring[-1])
        qc.h(coloring[-1])
        qc.x(coloring)
        qc.h(coloring)

    qc.measure(coloring, cr)
    return qc


# ============ Our MSB-Overflow Min Width oracle, unweighted mode ============
def msb_overflow_unweighted(N, edges, grover_iters):
    """Our full MSB-Overflow Min Width oracle, configured for the unweighted
    decision problem: every edge carries weight 1, target = |E|, so the
    MSB-Overflow flag triggers only when every edge is properly coloured.

    Uses the exact same `create_min_width_oracle` that produces Tables II,
    VI, VII; we only override the weights and target."""
    weighted = [(u, v, 1) for (u, v) in edges]
    target = len(edges)
    return create_min_width_oracle(N, K, weighted, target,
                                    grover_iterations=grover_iters)


# ============ Brute-force ground truth ======================================
def count_proper_colorings(N, edges):
    return sum(
        1 for colors in product(range(K), repeat=N)
        if all(colors[u] != colors[v] for (u, v) in edges)
    )


def optimal_grover_R(N, M):
    if M == 0:
        return 1
    return max(1, math.floor((math.pi / 4) * math.sqrt(K**N / M)))


# ============ Driver ========================================================
TOPOLOGIES = [
    ("Triangle",      3, [(0,1),(1,2),(0,2)]),
    ("Star S4",       4, [(0,1),(0,2),(0,3)]),
    ("Diag Square",   4, [(0,1),(1,2),(2,3),(0,3),(0,2)]),
    ("Path P5",       5, [(0,1),(1,2),(2,3),(3,4)]),
    ("Frustrated K4", 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]),
]


def measure(qc, edges, q=2):
    sim = AerSimulator()
    qc_t = transpile(qc, sim, basis_gates=BASIS, optimization_level=3)
    n_qubits = qc.num_qubits
    depth = qc_t.depth()
    result = sim.run(qc_t, shots=SHOTS).result()
    counts = result.get_counts()
    proper = 0
    for bs, c in counts.items():
        bs_clean = bs.replace(" ", "")[::-1]
        N = len(bs_clean) // q
        colors = []
        for i in range(N):
            chunk = bs_clean[i*q:(i+1)*q][::-1]
            colors.append(int(chunk, 2))
        if any(ci >= K for ci in colors):
            continue
        if all(colors[u] != colors[v] for (u, v) in edges):
            proper += c
    return n_qubits, depth, proper / SHOTS


def main():
    print("=== Unweighted head-to-head: Belletti orig vs our MSB-Overflow ===")
    print("(both run UNWEIGHTED proper-colouring problem, optimal R per topology)")
    print()
    print(f"{'Topology':<16} | {'(N,E)':<7} | {'M':<3} | {'R':<2} | "
          f"{'Belletti orig':<25} | {'MSB-Overflow (unweighted)'}")
    print(f"{'':<16} | {'':<7} | {'':<3} | {'':<2} | "
          f"{'Q':<3} {'D':<6} {'P_prop':<10} | {'Q':<3} {'D':<6} {'P_prop'}")
    print("-" * 100)
    for name, N, edges in TOPOLOGIES:
        M = count_proper_colorings(N, edges)
        R = optimal_grover_R(N, M)
        E = len(edges)

        qc_a = belletti_original(N, edges, R)
        Q_a, D_a, P_a = measure(qc_a, edges)

        qc_b = msb_overflow_unweighted(N, edges, R)
        Q_b, D_b, P_b = measure(qc_b, edges)

        print(f"{name:<16} | ({N},{E})   | {M:<3} | {R:<2} | "
              f"{Q_a:<3} {D_a:<6} {P_a*100:>6.2f}%   | "
              f"{Q_b:<3} {D_b:<6} {P_b*100:>6.2f}%")


if __name__ == "__main__":
    main()
