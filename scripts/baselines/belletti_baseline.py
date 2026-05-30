"""
belletti_baseline.py — Faithful Belletti-style unweighted oracle for resource comparison.

Implements the Boolean Grover oracle described in Belletti et al. 2025
(as characterised in our Section 1.3) and transpiles it on the same five
topologies used for Table 2 (`tab:results_n`). We report:

  - qubit width (Nq + |V| + |E| + 1)
  - transpiled depth on the same basis gate set as our MSB-Overflow oracle
    (cx, id, rz, sx, x; optimization_level = 3)

Belletti's oracle is unweighted: it phase-marks colourings whose every edge
satisfies C(u) != C(v) AND whose every node holds a valid colour (no phantom
state outside [0, K-1]). It cannot represent the MWCSP target T, so the table
in the paper only compares resource consumption per Grover iteration; we do not
attempt to compare solution quality on Frustrated K4 since Belletti returns
zero amplitude there by construction.

Run:  .venv/bin/python research_resources/scripts/baselines/belletti_baseline.py
"""
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

BASIS = ["cx", "id", "rz", "sx", "x"]


def belletti_oracle(N, K, edges, grover_iterations=1):
    """Build a Belletti-style Boolean Grover oracle on N nodes, K colours.

    Layout:
      - coloring register: Nq qubits (q = ceil(log2 K))
      - node_valid ancilla: |V| qubits, one per node, flagging non-phantom
      - edge_valid ancilla: |E| qubits, one per edge, flagging C(u) != C(v)
      - phase ancilla:       1 qubit, prepared in |-> for kickback

    Per Grover iteration:
      1. Generic Hadamard init H^otimes(Nq)        (only on first iteration)
      2. For each node, set node_valid[v] = 1 iff colour register is in [0, K-1].
         For K=3, q=2 this means flag if reg != |11>. We use one MCX (on the
         two qubits) controlled on the not-pattern to flip the ancilla,
         consistent with the per-node invalid checker described in Belletti
         (which costs O(q) Toffolis per node in their construction).
      3. For each edge (u, v), set edge_valid[e] = 1 iff C(u) != C(v).
         Standard CX-X-MCX-X pattern, output XOR-flipped to encode inequality.
      4. Phase kickback: CNOT-chain (one big MCX over all node_valid and
         edge_valid flags) flipping the phase ancilla iff every flag is set.
      5. Uncompute steps 2-3 in reverse order to disentangle ancillas.
      6. Diffusion: A H^otimes(Nq) ; X^otimes(Nq) ; multi-controlled-Z ; X^otimes(Nq) ; H^otimes(Nq).
    """
    q = math.ceil(math.log2(K)) if K > 1 else 1
    Nq = N * q
    E = len(edges)

    coloring  = QuantumRegister(Nq, "color")
    node_anc  = QuantumRegister(N, "nodev")
    edge_anc  = QuantumRegister(E, "edgev")
    phase_anc = QuantumRegister(1, "phase")
    cr = ClassicalRegister(Nq, "c")
    qc = QuantumCircuit(coloring, node_anc, edge_anc, phase_anc, cr)

    # Generic Hadamard init (Belletti's choice; populates phantoms when K is
    # not a power of 2)
    qc.h(coloring)
    qc.x(phase_anc[0]); qc.h(phase_anc[0])  # prepare |->

    # We treat colour value c >= K as the phantom set. For K=3, q=2 the only
    # phantom is |11>. We flip node_anc[v] when the node register is NOT a
    # phantom, i.e. node_anc[v] starts as 1 and is undone whenever the register
    # is exactly |11>.
    def per_node_valid(qc, node):
        qs = [coloring[node * q + j] for j in range(q)]
        # node_anc starts at |0>. Flip to |1> unconditionally, then
        # CCX on (qs all 1) -> flip back to |0>. End result: node_anc = 1 iff
        # not phantom.
        qc.x(node_anc[node])
        if q == 1:
            qc.cx(qs[0], node_anc[node])
        else:
            qc.mcx(qs, node_anc[node])

    def per_node_valid_inv(qc, node):
        qs = [coloring[node * q + j] for j in range(q)]
        if q == 1:
            qc.cx(qs[0], node_anc[node])
        else:
            qc.mcx(qs, node_anc[node])
        qc.x(node_anc[node])

    def per_edge_check(qc, e_idx, u, v):
        u_qs = [coloring[u * q + j] for j in range(q)]
        v_qs = [coloring[v * q + j] for j in range(q)]
        # XOR u into v
        for j in range(q):
            qc.cx(u_qs[j], v_qs[j])
            qc.x(v_qs[j])
        qc.mcx(v_qs, edge_anc[e_idx])
        for j in reversed(range(q)):
            qc.x(v_qs[j])
            qc.cx(u_qs[j], v_qs[j])
        qc.x(edge_anc[e_idx])  # invert: 1 iff inequality

    def per_edge_check_inv(qc, e_idx, u, v):
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

    for _ in range(grover_iterations):
        for v in range(N):
            per_node_valid(qc, v)
        for e_idx, (u, v, _w) in enumerate(edges):
            per_edge_check(qc, e_idx, u, v)
        # Phase kickback: AND of all flags -> flip phase ancilla
        all_flags = list(node_anc) + list(edge_anc)
        qc.mcx(all_flags, phase_anc[0])
        # Uncompute edges then nodes
        for e_idx, (u, v, _w) in reversed(list(enumerate(edges))):
            per_edge_check_inv(qc, e_idx, u, v)
        for v in range(N):
            per_node_valid_inv(qc, v)
        # Diffusion
        qc.h(coloring)
        qc.x(coloring)
        qc.h(coloring[-1])
        qc.mcx(coloring[:-1], coloring[-1])
        qc.h(coloring[-1])
        qc.x(coloring)
        qc.h(coloring)

    qc.measure(coloring, cr)
    return qc


def main():
    topologies = {
        "Triangle":         {"N": 3, "edges": [(0, 1, 1), (1, 2, 2), (0, 2, 3)]},
        "Star S4":          {"N": 4, "edges": [(0, 1, 4), (0, 2, 2), (0, 3, 5)]},
        "Diag Square":      {"N": 4, "edges": [(0, 1, 1), (1, 2, 2), (2, 3, 3), (0, 3, 4), (0, 2, 10)]},
        "Path P5":          {"N": 5, "edges": [(0, 1, 3), (1, 2, 7), (2, 3, 2), (3, 4, 6)]},
        "Frustrated K4":    {"N": 4, "edges": [(0, 1, 4), (0, 2, 5), (0, 3, 3), (1, 2, 6), (1, 3, 2), (2, 3, 4)]},
    }
    K = 3
    sim = AerSimulator()

    print(f"{'Topology':<16} | {'(N,E)':<7} | {'Qubits':<6} | {'Depth (untranspiled)':<22} | {'Depth (transpiled, opt=3)'}")
    print("-" * 100)
    for name, data in topologies.items():
        N, edges = data["N"], data["edges"]
        qc = belletti_oracle(N, K, edges, grover_iterations=1)
        qubits = qc.num_qubits
        d_pre = qc.depth()
        qc_t = transpile(qc, sim, basis_gates=BASIS, optimization_level=3)
        d_post = qc_t.depth()
        print(f"{name:<16} | ({N},{len(edges)})  | {qubits:<6} | {d_pre:<22} | {d_post}")


if __name__ == "__main__":
    main()
