"""
verify_table2.py — Ground-truth (qubits, depth) extractor for Table 2 (`tab:results_n`).

Copies the 3 oracle constructors from `tqe_experimental_results.py` (which can't be
imported because the source file uses IPython display macros at top level) and runs
them across the 5 benchmark topologies declared in the paper.

Outputs a single Markdown table that can be diffed against Section 4 Table 2.

Run:  .venv/bin/python verify_table2.py
"""
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


# ---------- Subcircuits (copied verbatim from tqe_experimental_results.py) ----------
def _create_superposition_subcircuit(M, n_qubits):
    n_required = int(np.ceil(np.log2(M)))
    if n_qubits < n_required:
        raise ValueError(f"Need {n_required} qubits for M={M}, got {n_qubits}.")
    qc = QuantumCircuit(n_qubits, name=f"Uniform({M})")
    if (M & (M - 1)) == 0:
        num_h = int(np.log2(M))
        qc.h(range(num_h))
        return qc
    l = [i for i, bit in enumerate(bin(M)[2:][::-1]) if bit == '1']
    k = len(l) - 1
    if k > 0:
        qc.x(l[1:])
    l0 = l[0]
    M0 = 2**l0
    if l0 > 0:
        qc.h(range(l0))
    if k > 0:
        theta0 = -2 * np.arccos(np.sqrt(M0 / M))
        qc.ry(theta0, l[1])
        qc.x(l[1])
        for target in range(l0, l[1]):
            qc.ch(l[1], target)
        qc.x(l[1])
    M_prev = M0
    for m in range(1, k):
        theta_m = -2 * np.arccos(np.sqrt(2**l[m] / (M - M_prev)))
        control, target = l[m], l[m+1]
        qc.x(control)
        qc.cry(theta_m, control, target)
        qc.x(control)
        qc.x(l[m+1])
        for t_h in range(l[m], l[m+1]):
            qc.ch(l[m+1], t_h)
        qc.x(l[m+1])
    return qc


def build_incrementor(counter_size):
    qc = QuantumCircuit(counter_size + 1, name="Inc")
    for target in reversed(range(1, counter_size + 1)):
        controls = list(range(0, target))
        if len(controls) == 1:
            qc.cx(controls[0], target)
        else:
            qc.mcx(controls, target)
    return qc


def add_constant(qc, W, control_qubit, counter_qubits):
    bin_W = bin(W)[2:][::-1]
    for k, bit in enumerate(bin_W):
        if bit == '1':
            inc_size = len(counter_qubits) - k
            inc = build_incrementor(inc_size)
            qs = [control_qubit] + counter_qubits[k:]
            qc.compose(inc, qubits=qs, inplace=True)


def dec_constant(qc, W, control_qubit, counter_qubits):
    bin_W = bin(W)[2:][::-1]
    for k, bit in reversed(list(enumerate(bin_W))):
        if bit == '1':
            inc_size = len(counter_qubits) - k
            dec = build_incrementor(inc_size).inverse()
            qs = [control_qubit] + counter_qubits[k:]
            qc.compose(dec, qubits=qs, inplace=True)


def belletti_auto_chunk_size(E):
    return math.ceil((1 + math.sqrt(8 * E - 7)) / 2)


# ---------- Min Width Oracle ----------
def create_min_width_oracle(N, K, weighted_edges, target_score, grover_iterations=1):
    q_per_node = math.ceil(math.log2(K)) if K > 1 else 1
    num_coloring_qubits = N * q_per_node
    max_score = sum(w for _, _, w in weighted_edges)
    counter_size = math.ceil(math.log2(max_score + 1)) + 1

    coloring = QuantumRegister(num_coloring_qubits, 'color')
    edge_anc = QuantumRegister(1, 'edge_anc')
    counter = QuantumRegister(counter_size, 'score')
    phase_anc = QuantumRegister(1, 'phase')
    cr = ClassicalRegister(num_coloring_qubits, 'c')
    qc = QuantumCircuit(coloring, edge_anc, counter, phase_anc, cr)

    state_prep = QuantumCircuit(num_coloring_qubits, name="StatePrep")
    for i in range(N):
        sub_circ = _create_superposition_subcircuit(K, q_per_node)
        if sub_circ.num_qubits > 0:
            state_prep.compose(sub_circ, qubits=list(range(i*q_per_node, (i+1)*q_per_node)), inplace=True)
    qc.compose(state_prep, qubits=coloring, inplace=True)
    qc.x(phase_anc[0])
    qc.h(phase_anc[0])

    for _ in range(grover_iterations):
        offset = (2**(counter_size - 1)) - target_score
        offset_bin = bin(offset)[2:].zfill(counter_size)
        for i, bit in enumerate(reversed(offset_bin)):
            if bit == '1':
                qc.x(counter[i])
        for u, v, w in weighted_edges:
            u_qs = [coloring[u*q_per_node + j] for j in range(q_per_node)]
            v_qs = [coloring[v*q_per_node + j] for j in range(q_per_node)]
            for j in range(q_per_node):
                qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
            qc.mcx(v_qs, edge_anc[0])
            for j in reversed(range(q_per_node)):
                qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
            qc.x(edge_anc[0])
            add_constant(qc, w, edge_anc[0], list(counter))
            qc.x(edge_anc[0])
            for j in range(q_per_node):
                qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
            qc.mcx(v_qs, edge_anc[0])
            for j in reversed(range(q_per_node)):
                qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
        qc.cx(counter[counter_size - 1], phase_anc[0])
        for u, v, w in reversed(weighted_edges):
            u_qs = [coloring[u*q_per_node + j] for j in range(q_per_node)]
            v_qs = [coloring[v*q_per_node + j] for j in range(q_per_node)]
            for j in range(q_per_node):
                qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
            qc.mcx(v_qs, edge_anc[0])
            for j in reversed(range(q_per_node)):
                qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
            qc.x(edge_anc[0])
            dec_constant(qc, w, edge_anc[0], list(counter))
            qc.x(edge_anc[0])
            for j in range(q_per_node):
                qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
            qc.mcx(v_qs, edge_anc[0])
            for j in reversed(range(q_per_node)):
                qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
        for i, bit in enumerate(reversed(offset_bin)):
            if bit == '1':
                qc.x(counter[i])
        qc.compose(state_prep.inverse(), qubits=coloring, inplace=True)
        qc.x(coloring)
        qc.h(coloring[-1]); qc.mcx(coloring[:-1], coloring[-1]); qc.h(coloring[-1])
        qc.x(coloring)
        qc.compose(state_prep, qubits=coloring, inplace=True)
    qc.measure(coloring, cr)
    return qc


# ---------- Min Depth Oracle ----------
def create_min_depth_oracle(N, K, weighted_edges, target_score, grover_iterations=1):
    q_per_node = math.ceil(math.log2(K)) if K > 1 else 1
    num_coloring_qubits = N * q_per_node
    E = len(weighted_edges)
    max_score = sum(w for _, _, w in weighted_edges)
    counter_size = math.ceil(math.log2(max_score + 1)) + 1

    coloring = QuantumRegister(num_coloring_qubits, 'color')
    edge_ancs = QuantumRegister(E, 'edge_ancs')
    counter = QuantumRegister(counter_size, 'score')
    phase_anc = QuantumRegister(1, 'phase')
    cr = ClassicalRegister(num_coloring_qubits, 'c')
    qc = QuantumCircuit(coloring, edge_ancs, counter, phase_anc, cr)

    state_prep = QuantumCircuit(num_coloring_qubits, name="StatePrep")
    for i in range(N):
        sub_circ = _create_superposition_subcircuit(K, q_per_node)
        if sub_circ.num_qubits > 0:
            state_prep.compose(sub_circ, qubits=list(range(i*q_per_node, (i+1)*q_per_node)), inplace=True)
    qc.compose(state_prep, qubits=coloring, inplace=True)
    qc.x(phase_anc[0])
    qc.h(phase_anc[0])

    for _ in range(grover_iterations):
        offset = (2**(counter_size - 1)) - target_score
        offset_bin = bin(offset)[2:].zfill(counter_size)
        for i, bit in enumerate(reversed(offset_bin)):
            if bit == '1':
                qc.x(counter[i])
        for i, (u, v, _) in enumerate(weighted_edges):
            u_qs = [coloring[u*q_per_node + j] for j in range(q_per_node)]
            v_qs = [coloring[v*q_per_node + j] for j in range(q_per_node)]
            for j in range(q_per_node):
                qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
            qc.mcx(v_qs, edge_ancs[i])
            for j in reversed(range(q_per_node)):
                qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
            qc.x(edge_ancs[i])
        for i, (_, _, w) in enumerate(weighted_edges):
            add_constant(qc, w, edge_ancs[i], list(counter))
        qc.cx(counter[counter_size - 1], phase_anc[0])
        for i, (_, _, w) in reversed(list(enumerate(weighted_edges))):
            dec_constant(qc, w, edge_ancs[i], list(counter))
        for i, (u, v, _) in reversed(list(enumerate(weighted_edges))):
            u_qs = [coloring[u*q_per_node + j] for j in range(q_per_node)]
            v_qs = [coloring[v*q_per_node + j] for j in range(q_per_node)]
            qc.x(edge_ancs[i])
            for j in range(q_per_node):
                qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
            qc.mcx(v_qs, edge_ancs[i])
            for j in reversed(range(q_per_node)):
                qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
        for i, bit in enumerate(reversed(offset_bin)):
            if bit == '1':
                qc.x(counter[i])
        qc.compose(state_prep.inverse(), qubits=coloring, inplace=True)
        qc.x(coloring)
        qc.h(coloring[-1]); qc.mcx(coloring[:-1], coloring[-1]); qc.h(coloring[-1])
        qc.x(coloring)
        qc.compose(state_prep, qubits=coloring, inplace=True)
    qc.measure(coloring, cr)
    return qc


# ---------- Balanced Oracle ----------
def create_balanced_oracle(N, K, weighted_edges, chunk_size, target_score, grover_iterations=1):
    q_per_node = math.ceil(math.log2(K)) if K > 1 else 1
    num_coloring_qubits = N * q_per_node
    E = len(weighted_edges)
    actual_chunk = min(chunk_size, E)
    max_score = sum(w for _, _, w in weighted_edges)
    counter_size = math.ceil(math.log2(max_score + 1)) + 1

    coloring = QuantumRegister(num_coloring_qubits, 'color')
    edge_ancs = QuantumRegister(actual_chunk, 'edge_ancs')
    counter = QuantumRegister(counter_size, 'score')
    phase_anc = QuantumRegister(1, 'phase')
    cr = ClassicalRegister(num_coloring_qubits, 'c')
    qc = QuantumCircuit(coloring, edge_ancs, counter, phase_anc, cr)

    state_prep = QuantumCircuit(num_coloring_qubits, name="StatePrep")
    for i in range(N):
        sub_circ = _create_superposition_subcircuit(K, q_per_node)
        if sub_circ.num_qubits > 0:
            state_prep.compose(sub_circ, qubits=list(range(i*q_per_node, (i+1)*q_per_node)), inplace=True)
    qc.compose(state_prep, qubits=coloring, inplace=True)
    qc.x(phase_anc[0])
    qc.h(phase_anc[0])

    chunks = [weighted_edges[i:i+actual_chunk] for i in range(0, E, actual_chunk)]

    for _ in range(grover_iterations):
        offset = (2**(counter_size - 1)) - target_score
        offset_bin = bin(offset)[2:].zfill(counter_size)
        for i, bit in enumerate(reversed(offset_bin)):
            if bit == '1':
                qc.x(counter[i])
        for chunk in chunks:
            for idx, (u, v, _) in enumerate(chunk):
                u_qs = [coloring[u*q_per_node + j] for j in range(q_per_node)]
                v_qs = [coloring[v*q_per_node + j] for j in range(q_per_node)]
                for j in range(q_per_node):
                    qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
                qc.mcx(v_qs, edge_ancs[idx])
                for j in reversed(range(q_per_node)):
                    qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
                qc.x(edge_ancs[idx])
            for idx, (_, _, w) in enumerate(chunk):
                add_constant(qc, w, edge_ancs[idx], list(counter))
            for idx in reversed(range(len(chunk))):
                u, v, _ = chunk[idx]
                u_qs = [coloring[u*q_per_node + j] for j in range(q_per_node)]
                v_qs = [coloring[v*q_per_node + j] for j in range(q_per_node)]
                qc.x(edge_ancs[idx])
                for j in range(q_per_node):
                    qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
                qc.mcx(v_qs, edge_ancs[idx])
                for j in reversed(range(q_per_node)):
                    qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
        qc.cx(counter[counter_size - 1], phase_anc[0])
        for chunk in reversed(chunks):
            for idx, (u, v, _) in enumerate(chunk):
                u_qs = [coloring[u*q_per_node + j] for j in range(q_per_node)]
                v_qs = [coloring[v*q_per_node + j] for j in range(q_per_node)]
                for j in range(q_per_node):
                    qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
                qc.mcx(v_qs, edge_ancs[idx])
                for j in reversed(range(q_per_node)):
                    qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
                qc.x(edge_ancs[idx])
            for idx in reversed(range(len(chunk))):
                _, _, w = chunk[idx]
                dec_constant(qc, w, edge_ancs[idx], list(counter))
            for idx in reversed(range(len(chunk))):
                u, v, _ = chunk[idx]
                u_qs = [coloring[u*q_per_node + j] for j in range(q_per_node)]
                v_qs = [coloring[v*q_per_node + j] for j in range(q_per_node)]
                qc.x(edge_ancs[idx])
                for j in range(q_per_node):
                    qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
                qc.mcx(v_qs, edge_ancs[idx])
                for j in reversed(range(q_per_node)):
                    qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
        for i, bit in enumerate(reversed(offset_bin)):
            if bit == '1':
                qc.x(counter[i])
        qc.compose(state_prep.inverse(), qubits=coloring, inplace=True)
        qc.x(coloring)
        qc.h(coloring[-1]); qc.mcx(coloring[:-1], coloring[-1]); qc.h(coloring[-1])
        qc.x(coloring)
        qc.compose(state_prep, qubits=coloring, inplace=True)
    qc.measure(coloring, cr)
    return qc


# ---------- Classical brute-force (for Target) ----------
def classical_best_score(N, K, edges):
    best = -1
    for cfg in range(K**N):
        colors = []
        v = cfg
        for _ in range(N):
            colors.append(v % K); v //= K
        s = sum(w for u, vv, w in edges if colors[u] != colors[vv])
        if s > best:
            best = s
    return best


# ---------- Benchmark topologies (copied from tqe_experimental_results.py) ----------
TOPOLOGIES = [
    ('Triangle',         3, 3, [(0,1,1),(1,2,2),(0,2,3)]),
    ('Star $S_4$',       4, 3, [(0,1,4),(0,2,2),(0,3,5)]),
    ('Diag Square',      4, 3, [(0,1,2),(1,2,2),(2,3,2),(3,0,2),(0,2,10)]),
    ('Path $P_5$',       5, 3, [(0,1,3),(1,2,7),(2,3,2),(3,4,6)]),
    ('Frustrated $K_4$', 4, 3, [(0,1,10),(0,2,10),(0,3,10),(1,2,10),(1,3,10),(2,3,2)]),
]

BASIS_GATES = ['cx', 'id', 'rz', 'sx', 'x']

def run(target_strategy: str, depth_kind: str):
    backend = AerSimulator()
    print(f"\n=== Target={target_strategy} | Depth={depth_kind} ===\n")
    print(f"{'Graph':<22} {'(N,E)':<8} {'W_max':<6} {'S':<3} {'C*':<4} "
          f"{'MW_q':<5} {'MW_d':<6} {'MD_q':<5} {'MD_d':<6} {'Bal_q':<6} {'Bal_d':<6}")
    print('-' * 95)

    for name, N, K, edges in TOPOLOGIES:
        E = len(edges)
        W_max = sum(w for _, _, w in edges)
        S = math.ceil(math.log2(W_max + 1)) + 1
        C = belletti_auto_chunk_size(E)
        if target_strategy == 'optimal':
            target = classical_best_score(N, K, edges)
        elif target_strategy == '75pct':
            target = max(1, math.floor(W_max * 0.75))
        else:
            raise ValueError(target_strategy)

        qc_mw  = create_min_width_oracle(N, K, edges, target, 1)
        qc_md  = create_min_depth_oracle(N, K, edges, target, 1)
        qc_bal = create_balanced_oracle(N, K, edges, C, target, 1)

        if depth_kind == 'raw':
            d_mw, d_md, d_bal = qc_mw.depth(), qc_md.depth(), qc_bal.depth()
        elif depth_kind == 'transpiled':
            t_mw  = transpile(qc_mw,  backend, basis_gates=BASIS_GATES, optimization_level=3)
            t_md  = transpile(qc_md,  backend, basis_gates=BASIS_GATES, optimization_level=3)
            t_bal = transpile(qc_bal, backend, basis_gates=BASIS_GATES, optimization_level=3)
            d_mw, d_md, d_bal = t_mw.depth(), t_md.depth(), t_bal.depth()
        else:
            raise ValueError(depth_kind)

        print(f"{name:<22} ({N},{E})    {W_max:<6} {S:<3} {C:<4} "
              f"{qc_mw.num_qubits:<5} {d_mw:<6} "
              f"{qc_md.num_qubits:<5} {d_md:<6} "
              f"{qc_bal.num_qubits:<6} {d_bal:<6}")


if __name__ == "__main__":
    # Cover all 4 combinations so we can match whichever the published Table 2 used.
    run('optimal', 'transpiled')
    run('optimal', 'raw')
    run('75pct',   'transpiled')
    run('75pct',   'raw')
    print("\nCompare against Table 2 (`tab:results_n`) in tqe_weighted.tex to pick the matching mode.")
