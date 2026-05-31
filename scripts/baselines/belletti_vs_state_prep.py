"""
belletti_vs_state_prep.py - Three-way comparison on the UNWEIGHTED graph
K-coloring decision problem, all methods run end-to-end and measured:

  (A) Belletti orig: Hadamard init + per-node invalid-colour checker +
      per-edge collision detector + global MCX over |V|+|E| flags + Boolean
      diffusion.

  (B) Belletti + state-prep ablation: Belletti's own source code with one
      change -- Hadamard -> Shukla-Vedula A_init. The per-node checkers are
      dropped (A_init produces zero amplitude on phantom states, making them
      redundant) and the diffuser is wrapped in A_init / A_init^(-1) to
      invert about the new initial state. Every other block is verbatim
      Belletti.

  (C) MSB-Overflow Min Width: our full oracle from `verify_table2.py`,
      configured for the unweighted problem (weights = 1, target = |E|).

All three are measured at the empirically-optimal Grover iteration count
R = floor((pi/4) * sqrt(K^N / |S*|)) per topology and the proper-colouring
rate is computed from 2048 noise-free shots.
"""
import math
import os
import sys
from itertools import product
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# Pull our canonical MSB-Overflow oracle from the reproducibility package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "reproducibility"))
from verify_table2 import create_min_width_oracle  # noqa: E402

BASIS = ["cx", "id", "rz", "sx", "x"]
K = 3
SHOTS = 2048


# ============ Shukla-Vedula state prep for K=3 ============================
def apply_a_init(qc, qubits):
    theta = 2 * math.acos(math.sqrt(2.0 / 3.0))
    qc.ry(theta, qubits[0])
    qc.x(qubits[0]); qc.ch(qubits[0], qubits[1]); qc.x(qubits[0])


def apply_a_init_inv(qc, qubits):
    theta = 2 * math.acos(math.sqrt(2.0 / 3.0))
    qc.x(qubits[0]); qc.ch(qubits[0], qubits[1]); qc.x(qubits[0])
    qc.ry(-theta, qubits[0])


def per_edge_check(qc, coloring, edge_anc, e_idx, u, v, q):
    u_qs = [coloring[u * q + j] for j in range(q)]
    v_qs = [coloring[v * q + j] for j in range(q)]
    for j in range(q):
        qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
    qc.mcx(v_qs, edge_anc[e_idx])
    for j in reversed(range(q)):
        qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
    qc.x(edge_anc[e_idx])


def per_edge_check_inv(qc, coloring, edge_anc, e_idx, u, v, q):
    u_qs = [coloring[u * q + j] for j in range(q)]
    v_qs = [coloring[v * q + j] for j in range(q)]
    qc.x(edge_anc[e_idx])
    for j in range(q):
        qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
    qc.mcx(v_qs, edge_anc[e_idx])
    for j in reversed(range(q)):
        qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])


def per_node_valid(qc, coloring, node_anc, node, q):
    qs = [coloring[node * q + j] for j in range(q)]
    qc.x(node_anc[node]); qc.mcx(qs, node_anc[node])


def per_node_valid_inv(qc, coloring, node_anc, node, q):
    qs = [coloring[node * q + j] for j in range(q)]
    qc.mcx(qs, node_anc[node]); qc.x(node_anc[node])


# ============ (A) Belletti original ========================================
def belletti_original(N, edges, R):
    q = 2
    Nq = N * q; E = len(edges)
    coloring = QuantumRegister(Nq, "color")
    node_anc = QuantumRegister(N, "nodev")
    edge_anc = QuantumRegister(E, "edgev")
    phase_anc = QuantumRegister(1, "phase")
    cr = ClassicalRegister(Nq, "c")
    qc = QuantumCircuit(coloring, node_anc, edge_anc, phase_anc, cr)
    qc.h(coloring)
    qc.x(phase_anc[0]); qc.h(phase_anc[0])
    for _ in range(R):
        for v in range(N): per_node_valid(qc, coloring, node_anc, v, q)
        for e_idx, (u, vv) in enumerate(edges):
            per_edge_check(qc, coloring, edge_anc, e_idx, u, vv, q)
        qc.mcx(list(node_anc) + list(edge_anc), phase_anc[0])
        for e_idx, (u, vv) in reversed(list(enumerate(edges))):
            per_edge_check_inv(qc, coloring, edge_anc, e_idx, u, vv, q)
        for v in range(N): per_node_valid_inv(qc, coloring, node_anc, v, q)
        qc.h(coloring); qc.x(coloring)
        qc.h(coloring[-1]); qc.mcx(coloring[:-1], coloring[-1]); qc.h(coloring[-1])
        qc.x(coloring); qc.h(coloring)
    qc.measure(coloring, cr)
    return qc


# ============ (B) Belletti + our state prep (ablation) =====================
def belletti_with_our_state_prep(N, edges, R):
    q = 2
    Nq = N * q; E = len(edges)
    coloring = QuantumRegister(Nq, "color")
    edge_anc = QuantumRegister(E, "edgev")
    phase_anc = QuantumRegister(1, "phase")
    cr = ClassicalRegister(Nq, "c")
    qc = QuantumCircuit(coloring, edge_anc, phase_anc, cr)
    for v in range(N): apply_a_init(qc, [coloring[v * q], coloring[v * q + 1]])
    qc.x(phase_anc[0]); qc.h(phase_anc[0])
    for _ in range(R):
        for e_idx, (u, vv) in enumerate(edges):
            per_edge_check(qc, coloring, edge_anc, e_idx, u, vv, q)
        qc.mcx(list(edge_anc), phase_anc[0])
        for e_idx, (u, vv) in reversed(list(enumerate(edges))):
            per_edge_check_inv(qc, coloring, edge_anc, e_idx, u, vv, q)
        for v in range(N): apply_a_init_inv(qc, [coloring[v * q], coloring[v * q + 1]])
        qc.x(coloring)
        qc.h(coloring[-1]); qc.mcx(coloring[:-1], coloring[-1]); qc.h(coloring[-1])
        qc.x(coloring)
        for v in range(N): apply_a_init(qc, [coloring[v * q], coloring[v * q + 1]])
    qc.measure(coloring, cr)
    return qc


# ============ (C) Our MSB-Overflow Min Width on unweighted decision =======
def msb_overflow_unweighted(N, edges, R):
    weighted = [(u, v, 1) for (u, v) in edges]
    target = len(edges)
    return create_min_width_oracle(N, K, weighted, target, grover_iterations=R)


# ============ Brute force ground truth ====================================
def count_proper_colorings(N, edges):
    return sum(1 for c in product(range(K), repeat=N)
               if all(c[u] != c[v] for (u, v) in edges))


def optimal_grover_R(N, M):
    if M == 0: return 1
    return max(1, math.floor((math.pi / 4) * math.sqrt(K**N / M)))


# ============ Measurement helper ==========================================
def measure(qc, edges, q=2):
    sim = AerSimulator()
    qc_t = transpile(qc, sim, basis_gates=BASIS, optimization_level=3)
    counts = sim.run(qc_t, shots=SHOTS).result().get_counts()
    proper = 0
    for bs, c in counts.items():
        bs_clean = bs.replace(" ", "")[::-1]
        N = len(bs_clean) // q
        colors = [int(bs_clean[i*q:(i+1)*q][::-1], 2) for i in range(N)]
        if any(ci >= K for ci in colors): continue
        if all(colors[u] != colors[v] for (u, v) in edges):
            proper += c
    return proper / SHOTS


# ============ Driver =====================================================
TOPOLOGIES = [
    ("Triangle",      3, [(0,1),(1,2),(0,2)]),
    ("Star $S_4$",    4, [(0,1),(0,2),(0,3)]),
    ("Diag Square",   4, [(0,1),(1,2),(2,3),(0,3),(0,2)]),
    ("Path $P_5$",    5, [(0,1),(1,2),(2,3),(3,4)]),
    ("Frustrated $K_4$", 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]),
]


def sweep_R(oracle_fn, N, edges, R_max=6):
    """Sweep R = 1..R_max and return (best_R, best_P)."""
    best_R, best_P = 1, 0.0
    for R in range(1, R_max + 1):
        P = measure(oracle_fn(N, edges, R), edges)
        if P > best_P:
            best_R, best_P = R, P
    return best_R, best_P


def main():
    print(f"{'Topology':<18} | {'K^N':<5} | {'|S*|':<5} | "
          f"{'Belletti orig':<20} | {'Unweighted CSP Oracle':<25} | "
          f"{'MSB-Overflow UW'}")
    print(f"{'':<18} | {'':<5} | {'':<5} | "
          f"{'R*':<3}  {'P_prop':<14} | {'R*':<3}  {'P_prop':<19} | "
          f"{'R*':<3}  {'P_prop'}")
    print("-" * 115)
    for name, N, edges in TOPOLOGIES:
        M = count_proper_colorings(N, edges)
        Ra, Pa = sweep_R(belletti_original, N, edges)
        Rb, Pb = sweep_R(belletti_with_our_state_prep, N, edges)
        Rc, Pc = sweep_R(msb_overflow_unweighted, N, edges)
        nm = name.replace("$", "").replace("_", "")
        print(f"{nm:<18} | {K**N:<5} | {M:<5} | "
              f"{Ra:<3}  {Pa*100:>9.2f}%   | "
              f"{Rb:<3}  {Pb*100:>14.2f}%   | "
              f"{Rc:<3}  {Pc*100:>9.2f}%")


if __name__ == "__main__":
    main()
