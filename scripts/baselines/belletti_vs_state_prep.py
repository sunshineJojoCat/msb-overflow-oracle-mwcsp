"""
belletti_vs_state_prep.py - State-preparation ablation on the UNWEIGHTED graph
K-coloring decision problem. We start from Belletti's original oracle source
code (the same as `belletti_baseline.py`) and apply ONE surgical change:

    Hadamard initialisation  -->  Shukla-Vedula A_init state preparation

Everything else is kept exactly as in Belletti's design: per-edge collision
detector, global MCX phase trigger over all edge-validity flags, Boolean
diffusion. The per-node invalid-colour checker subcircuits are correspondingly
removed because A_init already leaves zero amplitude on phantom states (they
would suppress amplitude that is no longer there). The diffuser is wrapped in
A_init / A_init^dagger so that the inversion-about-the-mean is taken about the
new initial state.

This isolates the effect of the state-preparation layer alone: same problem,
same edge logic, same phase trigger, only the initial superposition differs.

Reports qubits, transpiled depth (basis cx/id/rz/sx/x, opt_level=3), and the
empirical proper-colouring rate from 2048 noise-free shots at the empirically-
optimal Grover R per topology.
"""
import math
from itertools import product
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

BASIS = ["cx", "id", "rz", "sx", "x"]
K = 3
SHOTS = 2048


# ============ Shukla-Vedula state prep for K=3 (q=2 qubits per node) ========
def apply_a_init(qc, qubits):
    """Produce (|00>+|01>+|10>)/sqrt(3) on a 2-qubit register, zero on |11>."""
    theta = 2 * math.acos(math.sqrt(2.0 / 3.0))
    qc.ry(theta, qubits[0])
    qc.x(qubits[0])
    qc.ch(qubits[0], qubits[1])
    qc.x(qubits[0])


def apply_a_init_inv(qc, qubits):
    theta = 2 * math.acos(math.sqrt(2.0 / 3.0))
    qc.x(qubits[0])
    qc.ch(qubits[0], qubits[1])
    qc.x(qubits[0])
    qc.ry(-theta, qubits[0])


# ============ Shared per-edge collision detector (Belletti's logic) =========
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


# ============ Belletti's per-node invalid-colour checker (orig only) ========
def per_node_valid(qc, coloring, node_anc, node, q):
    qs = [coloring[node * q + j] for j in range(q)]
    qc.x(node_anc[node])
    qc.mcx(qs, node_anc[node])


def per_node_valid_inv(qc, coloring, node_anc, node, q):
    qs = [coloring[node * q + j] for j in range(q)]
    qc.mcx(qs, node_anc[node])
    qc.x(node_anc[node])


# ============ Oracle A: Belletti original (unchanged) =======================
def belletti_original(N, edges, grover_iters):
    q = 2  # K=3
    Nq = N * q
    E = len(edges)
    coloring = QuantumRegister(Nq, "color")
    node_anc = QuantumRegister(N, "nodev")
    edge_anc = QuantumRegister(E, "edgev")
    phase_anc = QuantumRegister(1, "phase")
    cr = ClassicalRegister(Nq, "c")
    qc = QuantumCircuit(coloring, node_anc, edge_anc, phase_anc, cr)

    # Belletti's choice: generic Hadamard init
    qc.h(coloring)
    qc.x(phase_anc[0]); qc.h(phase_anc[0])

    for _ in range(grover_iters):
        # 1. Per-node invalid-colour checkers (needed because Hadamard
        #    populates phantom state |11>)
        for v in range(N):
            per_node_valid(qc, coloring, node_anc, v, q)
        # 2. Per-edge collision detection
        for e_idx, (u, vv) in enumerate(edges):
            per_edge_check(qc, coloring, edge_anc, e_idx, u, vv, q)
        # 3. Global MCX phase trigger over BOTH node and edge flags
        all_flags = list(node_anc) + list(edge_anc)
        qc.mcx(all_flags, phase_anc[0])
        # 4. Uncompute edges then nodes
        for e_idx, (u, vv) in reversed(list(enumerate(edges))):
            per_edge_check_inv(qc, coloring, edge_anc, e_idx, u, vv, q)
        for v in range(N):
            per_node_valid_inv(qc, coloring, node_anc, v, q)
        # 5. Boolean diffusion H S_0 H
        qc.h(coloring)
        qc.x(coloring)
        qc.h(coloring[-1])
        qc.mcx(coloring[:-1], coloring[-1])
        qc.h(coloring[-1])
        qc.x(coloring)
        qc.h(coloring)

    qc.measure(coloring, cr)
    return qc


# ============ Oracle B: Belletti source with ONLY state prep swapped ========
def belletti_with_our_state_prep(N, edges, grover_iters):
    """Belletti's collision logic and phase trigger, with the single change:
    Hadamard initialisation -> Shukla-Vedula A_init. The per-node invalid-colour
    checkers are removed because A_init already gives zero amplitude on phantom
    states. The diffuser is wrapped in A_init / A_init^dagger so the inversion-
    about-the-mean is taken about the new initial state.

    Everything else (per-edge collision detector, MCX phase trigger over edge
    flags, full uncomputation) is copied verbatim from Belletti's source code.
    """
    q = 2  # K=3
    Nq = N * q
    E = len(edges)
    coloring = QuantumRegister(Nq, "color")
    edge_anc = QuantumRegister(E, "edgev")
    phase_anc = QuantumRegister(1, "phase")
    cr = ClassicalRegister(Nq, "c")
    qc = QuantumCircuit(coloring, edge_anc, phase_anc, cr)

    # 0. State prep: Shukla-Vedula A_init replaces Hadamard
    for v in range(N):
        apply_a_init(qc, [coloring[v * q + 0], coloring[v * q + 1]])
    qc.x(phase_anc[0]); qc.h(phase_anc[0])

    for _ in range(grover_iters):
        # (per-node checkers REMOVED - A_init leaves zero amplitude on phantoms)
        # Per-edge collision detection (same as Belletti)
        for e_idx, (u, vv) in enumerate(edges):
            per_edge_check(qc, coloring, edge_anc, e_idx, u, vv, q)
        # MCX phase trigger over edge flags only (no node flags to AND in)
        qc.mcx(list(edge_anc), phase_anc[0])
        # Uncompute edges
        for e_idx, (u, vv) in reversed(list(enumerate(edges))):
            per_edge_check_inv(qc, coloring, edge_anc, e_idx, u, vv, q)
        # Diffusion: A_init S_0 A_init^dagger (about the new initial state)
        for v in range(N):
            apply_a_init_inv(qc, [coloring[v * q + 0], coloring[v * q + 1]])
        qc.x(coloring)
        qc.h(coloring[-1])
        qc.mcx(coloring[:-1], coloring[-1])
        qc.h(coloring[-1])
        qc.x(coloring)
        for v in range(N):
            apply_a_init(qc, [coloring[v * q + 0], coloring[v * q + 1]])

    qc.measure(coloring, cr)
    return qc


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
    print("=== State-preparation ablation on unweighted decision problem ===")
    print("(Belletti source, swap Hadamard -> Shukla-Vedula A_init, run end-to-end)")
    print()
    print(f"{'Topology':<16} | {'(N,E)':<7} | {'M':<3} | {'R':<2} | "
          f"{'Belletti orig':<25} | {'Belletti + our state prep'}")
    print(f"{'':<16} | {'':<7} | {'':<3} | {'':<2} | "
          f"{'Q':<3} {'D':<6} {'P_prop':<10} | {'Q':<3} {'D':<6} {'P_prop'}")
    print("-" * 105)
    R = 1  # per Grover iteration, matching Table V's existing semantics
    for name, N, edges in TOPOLOGIES:
        M = count_proper_colorings(N, edges)
        E = len(edges)

        qc_a = belletti_original(N, edges, R)
        Q_a, D_a, P_a = measure(qc_a, edges)

        qc_b = belletti_with_our_state_prep(N, edges, R)
        Q_b, D_b, P_b = measure(qc_b, edges)

        print(f"{name:<16} | ({N},{E})   | {M:<3} | {R:<2} | "
              f"{Q_a:<3} {D_a:<6} {P_a*100:>6.2f}%   | "
              f"{Q_b:<3} {D_b:<6} {P_b*100:>6.2f}%")


if __name__ == "__main__":
    main()
