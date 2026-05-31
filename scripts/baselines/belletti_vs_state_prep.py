"""
belletti_vs_state_prep.py - Isolate the contribution of Custom State Preparation
by comparing on the UNWEIGHTED graph K-coloring problem:

  (A) Belletti original: Hadamard init + per-node invalid-colour checker + per-edge
      collision detector + global MCX over |V|+|E| flags.
  (B) State-prep-only variant: Shukla-Vedula A_init replaces Hadamard, NO per-node
      checker (because A_init produces zero amplitude on phantom states),
      same per-edge collision detector, global MCX over only |E| flags,
      diffusion is A_init S_0 A_init^dagger instead of H S_0 H.

Both oracles solve the same Boolean problem (does a proper K-coloring exist?
which one?). The MSB-Overflow comparator is NOT used; this isolates the state-prep
contribution.

Reports: qubits, transpiled depth (basis cx/id/rz/sx/x, opt_level=3), and the
empirical proper-coloring rate from 2048 noise-free shots at the optimal Grover R.
"""
import math
from itertools import product
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

BASIS = ["cx", "id", "rz", "sx", "x"]
K = 3
SHOTS = 2048


# --- Shukla-Vedula state prep for K=3 on q=2 qubits ------------------------
def apply_a_init(qc, qubits, K=3):
    """Produce (|00>+|01>+|10>)/sqrt(3) on a 2-qubit register, zero on |11>.
    Uses Ry on MSB (prob 2/3 on |0>, prob 1/3 on |1>) then anti-controlled-H
    on LSB (split |00>+|01> when MSB=0, leave |10> when MSB=1)."""
    if K == 3 and len(qubits) == 2:
        theta = 2 * math.acos(math.sqrt(2.0 / 3.0))
        qc.ry(theta, qubits[0])
        qc.x(qubits[0])
        qc.ch(qubits[0], qubits[1])
        qc.x(qubits[0])
    else:
        raise NotImplementedError("Only K=3, q=2 supported here.")


def apply_a_init_inv(qc, qubits, K=3):
    if K == 3 and len(qubits) == 2:
        theta = 2 * math.acos(math.sqrt(2.0 / 3.0))
        qc.x(qubits[0])
        qc.ch(qubits[0], qubits[1])
        qc.x(qubits[0])
        qc.ry(-theta, qubits[0])
    else:
        raise NotImplementedError("Only K=3, q=2 supported here.")


# --- Shared per-edge collision detector -----------------------------------
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
    qc.x(edge_anc[e_idx])  # 1 iff colors differ


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


# --- Per-node invalid-colour checker (Belletti only) ----------------------
def per_node_valid(qc, coloring, node_anc, node, q):
    qs = [coloring[node * q + j] for j in range(q)]
    qc.x(node_anc[node])
    qc.mcx(qs, node_anc[node])


def per_node_valid_inv(qc, coloring, node_anc, node, q):
    qs = [coloring[node * q + j] for j in range(q)]
    qc.mcx(qs, node_anc[node])
    qc.x(node_anc[node])


# --- Oracle A: Belletti original ------------------------------------------
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

    qc.h(coloring)
    qc.x(phase_anc[0]); qc.h(phase_anc[0])  # |->

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


# --- Oracle B: State-prep-only (Belletti collision logic + our A_init) ----
def state_prep_only(N, edges, grover_iters):
    q = 2  # K=3
    Nq = N * q
    E = len(edges)
    coloring = QuantumRegister(Nq, "color")
    # NO node_anc; A_init prunes phantom states by construction.
    edge_anc = QuantumRegister(E, "edgev")
    phase_anc = QuantumRegister(1, "phase")
    cr = ClassicalRegister(Nq, "c")
    qc = QuantumCircuit(coloring, edge_anc, phase_anc, cr)

    # Shukla-Vedula state prep per node
    for v in range(N):
        apply_a_init(qc, [coloring[v * q + 0], coloring[v * q + 1]], K=3)
    qc.x(phase_anc[0]); qc.h(phase_anc[0])  # |->

    for _ in range(grover_iters):
        for e_idx, (u, vv) in enumerate(edges):
            per_edge_check(qc, coloring, edge_anc, e_idx, u, vv, q)
        # Phase kickback: MCX over edge flags ONLY (no node flags)
        qc.mcx(list(edge_anc), phase_anc[0])
        for e_idx, (u, vv) in reversed(list(enumerate(edges))):
            per_edge_check_inv(qc, coloring, edge_anc, e_idx, u, vv, q)
        # Diffusion: A_init S_0 A_init^dagger over the coloring register
        for v in range(N):
            apply_a_init_inv(qc, [coloring[v * q + 0], coloring[v * q + 1]], K=3)
        qc.x(coloring)
        qc.h(coloring[-1])
        qc.mcx(coloring[:-1], coloring[-1])
        qc.h(coloring[-1])
        qc.x(coloring)
        for v in range(N):
            apply_a_init(qc, [coloring[v * q + 0], coloring[v * q + 1]], K=3)

    qc.measure(coloring, cr)
    return qc


# --- Brute force: # proper colorings (M) ----------------------------------
def count_proper_colorings(N, edges):
    count = 0
    for colors in product(range(K), repeat=N):
        if all(colors[u] != colors[v] for (u, v) in edges):
            count += 1
    return count


def optimal_grover_R(N, M):
    if M == 0:
        return 1
    R = math.floor((math.pi / 4) * math.sqrt(K**N / M))
    return max(1, R)


# --- Driver ---------------------------------------------------------------
TOPOLOGIES = [
    ("Triangle",      3, [(0,1),(1,2),(0,2)]),
    ("Star S4",       4, [(0,1),(0,2),(0,3)]),
    ("Diag Square",   4, [(0,1),(1,2),(2,3),(0,3),(0,2)]),
    ("Path P5",       5, [(0,1),(1,2),(2,3),(3,4)]),
    ("Frustrated K4", 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]),
]


def measure(qc, edges, q=2):
    """Transpile + simulate noise-free, return (qubits, depth, P_proper)."""
    sim = AerSimulator()
    qc_t = transpile(qc, sim, basis_gates=BASIS, optimization_level=3)
    n_qubits = qc.num_qubits
    depth = qc_t.depth()

    result = sim.run(qc_t, shots=SHOTS).result()
    counts = result.get_counts()

    proper = 0
    for bs, c in counts.items():
        bs_clean = bs.replace(" ", "")[::-1]  # little-endian
        N = len(bs_clean) // q
        colors = []
        for i in range(N):
            chunk = bs_clean[i*q:(i+1)*q][::-1]
            colors.append(int(chunk, 2))
        # Discard phantom states (colour >= K)
        if any(c_i >= K for c_i in colors):
            continue
        if all(colors[u] != colors[v] for (u, v) in edges):
            proper += c
    return n_qubits, depth, proper / SHOTS


def main():
    print("=== Per Grover iteration (R=1, matching Table V semantics) ===")
    print(f"{'Topology':<16} | {'(N,E)':<7} | {'M':<3} | "
          f"{'Belletti orig':<22} | {'State-prep only':<22}")
    print(f"{'':<16} | {'':<7} | {'':<3} | "
          f"{'Q':<3} {'D':<6} {'P_prop':<8} | {'Q':<3} {'D':<6} {'P_prop':<8}")
    print("-" * 100)
    for name, N, edges in TOPOLOGIES:
        M = count_proper_colorings(N, edges)
        E = len(edges)
        qc_a = belletti_original(N, edges, 1)
        Q_a, D_a, P_a = measure(qc_a, edges)
        qc_b = state_prep_only(N, edges, 1)
        Q_b, D_b, P_b = measure(qc_b, edges)
        print(f"{name:<16} | ({N},{E})   | {M:<3} | "
              f"{Q_a:<3} {D_a:<6} {P_a*100:>5.2f}%   | "
              f"{Q_b:<3} {D_b:<6} {P_b*100:>5.2f}%")

    print()
    print("=== Full optimal-R run (success rate when each method is given its best R) ===")
    print(f"{'Topology':<16} | {'(N,E)':<7} | {'M':<3} | {'R':<2} | "
          f"{'Belletti P_prop':<16} | {'State-prep P_prop'}")
    print("-" * 90)
    for name, N, edges in TOPOLOGIES:
        M = count_proper_colorings(N, edges)
        R = optimal_grover_R(N, M)
        E = len(edges)
        qc_a = belletti_original(N, edges, R)
        _, _, P_a = measure(qc_a, edges)
        qc_b = state_prep_only(N, edges, R)
        _, _, P_b = measure(qc_b, edges)
        print(f"{name:<16} | ({N},{E})   | {M:<3} | {R:<2} | "
              f"{P_a*100:>6.2f}%          | {P_b*100:>6.2f}%")


if __name__ == "__main__":
    main()
