#!/usr/bin/env python3
"""
Generate ALL publication figures for the IEEE TQE Manuscript.
Run: python generate_all_figures.py
Requires: matplotlib, numpy, networkx, qiskit, qiskit_aer
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# Try importing Qiskit for circuit figures
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("WARNING: Qiskit not found. Circuit diagrams and simulation figures will use mock data.")

# ============================================================
# IEEE-quality plot configuration
# ============================================================
# NOTE ON LEGIBILITY (IEEE TQE resubmission fix):
# Figures are now rendered at their true on-page width (single column ~3.36 in,
# double column ~6.9 in for the dense figures) so that the vector PDF is NOT
# scaled down by \includegraphics. With the scale factor at ~1, the nominal
# point sizes below print at face value, keeping all labels at >= ~8 pt
# (comparable to the figure-caption font). All figures are exported as vector
# PDF rather than bitmap PNG.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ============================================================
# ORACLE FUNCTIONS (copied from comparision_weighted.py)
# ============================================================
def _create_superposition_subcircuit(M, n_qubits):
    n_required = int(np.ceil(np.log2(M)))
    if n_qubits < n_required: raise ValueError(f"Need {n_required} qubits for M={M}, got {n_qubits}.")
    qc = QuantumCircuit(n_qubits, name=f"Uniform({M})")
    if (M & (M - 1)) == 0:
        qc.h(range(int(np.log2(M))))
        return qc
    l = [i for i, bit in enumerate(bin(M)[2:][::-1]) if bit == '1']
    k = len(l) - 1
    if k > 0: qc.x(l[1:])
    l0 = l[0]; M0 = 2**l0
    if l0 > 0: qc.h(range(l0))
    if k > 0:
        theta0 = -2 * np.arccos(np.sqrt(M0 / M))
        qc.ry(theta0, l[1]); qc.x(l[1])
        for target in range(l0, l[1]): qc.ch(l[1], target)
        qc.x(l[1])
    M_prev = M0
    for m in range(1, k):
        theta_m = -2 * np.arccos(np.sqrt(2**l[m] / (M - M_prev)))
        control, target_q = l[m], l[m+1]
        qc.x(control); qc.cry(theta_m, control, target_q); qc.x(control)
        qc.x(l[m+1])
        for t_h in range(l[m], l[m+1]): qc.ch(l[m+1], t_h)
        qc.x(l[m+1])
    return qc

def build_incrementor(counter_size):
    qc = QuantumCircuit(counter_size + 1, name="Inc")
    for target in reversed(range(1, counter_size + 1)):
        controls = list(range(0, target))
        if len(controls) == 1: qc.cx(controls[0], target)
        else: qc.mcx(controls, target)
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

def create_weighted_balanced_oracle(N, K, weighted_edges, chunk_size, target_score, grover_iterations=1):
    q_per_node = math.ceil(math.log2(K)) if K > 1 else 1
    num_coloring_qubits, E = N * q_per_node, len(weighted_edges)
    actual_chunk = min(chunk_size, E)
    max_score = sum(w for _, _, w in weighted_edges)
    counter_size = math.ceil(math.log2(max_score + 1)) + 1

    coloring = QuantumRegister(num_coloring_qubits, 'color')
    edge_ancs = QuantumRegister(actual_chunk, 'edge')
    counter = QuantumRegister(counter_size, 'score')
    phase_anc = QuantumRegister(1, 'phase')
    cr = ClassicalRegister(num_coloring_qubits, 'meas')
    qc = QuantumCircuit(coloring, edge_ancs, counter, phase_anc, cr)

    state_prep = QuantumCircuit(num_coloring_qubits)
    for i in range(N):
        sub_circ = _create_superposition_subcircuit(K, q_per_node)
        if sub_circ.num_qubits > 0: state_prep.compose(sub_circ, qubits=list(range(i*q_per_node, (i+1)*q_per_node)), inplace=True)
    qc.compose(state_prep, qubits=coloring, inplace=True)
    qc.x(phase_anc[0]); qc.h(phase_anc[0])

    chunks = [weighted_edges[i:i+actual_chunk] for i in range(0, E, actual_chunk)]

    for _ in range(grover_iterations):
        offset = (2**(counter_size - 1)) - target_score
        offset_bin = bin(offset)[2:].zfill(counter_size)
        for i, bit in enumerate(reversed(offset_bin)):
            if bit == '1': qc.x(counter[i])

        for chunk in chunks:
            for idx, (u, v, _) in enumerate(chunk):
                u_qs = [coloring[u*q_per_node + j] for j in range(q_per_node)]
                v_qs = [coloring[v*q_per_node + j] for j in range(q_per_node)]
                for j in range(q_per_node): qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
                qc.mcx(v_qs, edge_ancs[idx])
                for j in reversed(range(q_per_node)): qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
                qc.x(edge_ancs[idx])
            for idx, (_, _, w) in enumerate(chunk): add_constant(qc, w, edge_ancs[idx], list(counter))
            for idx in reversed(range(len(chunk))):
                u, v, _ = chunk[idx]
                u_qs = [coloring[u*q_per_node + j] for j in range(q_per_node)]
                v_qs = [coloring[v*q_per_node + j] for j in range(q_per_node)]
                qc.x(edge_ancs[idx])
                for j in range(q_per_node): qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
                qc.mcx(v_qs, edge_ancs[idx])
                for j in reversed(range(q_per_node)): qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])

        qc.cx(counter[counter_size - 1], phase_anc[0])

        for chunk in reversed(chunks):
            for idx, (u, v, _) in enumerate(chunk):
                u_qs = [coloring[u*q_per_node + j] for j in range(q_per_node)]
                v_qs = [coloring[v*q_per_node + j] for j in range(q_per_node)]
                for j in range(q_per_node): qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
                qc.mcx(v_qs, edge_ancs[idx])
                for j in reversed(range(q_per_node)): qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])
                qc.x(edge_ancs[idx])
            for idx in reversed(range(len(chunk))):
                _, _, w = chunk[idx]
                dec_constant(qc, w, edge_ancs[idx], list(counter))
            for idx in reversed(range(len(chunk))):
                u, v, _ = chunk[idx]
                u_qs = [coloring[u*q_per_node + j] for j in range(q_per_node)]
                v_qs = [coloring[v*q_per_node + j] for j in range(q_per_node)]
                qc.x(edge_ancs[idx])
                for j in range(q_per_node): qc.cx(u_qs[j], v_qs[j]); qc.x(v_qs[j])
                qc.mcx(v_qs, edge_ancs[idx])
                for j in reversed(range(q_per_node)): qc.x(v_qs[j]); qc.cx(u_qs[j], v_qs[j])

        for i, bit in enumerate(reversed(offset_bin)):
            if bit == '1': qc.x(counter[i])

        qc.compose(state_prep.inverse(), qubits=coloring, inplace=True)
        qc.x(coloring); qc.h(coloring[-1]); qc.mcx(coloring[:-1], coloring[-1]); qc.h(coloring[-1]); qc.x(coloring)
        qc.compose(state_prep, qubits=coloring, inplace=True)

    qc.measure(coloring, cr)
    return qc


# ============================================================
# FIGURE 1: Before & After Circuit Comparison
# ============================================================
def generate_fig1_before_after():
    print("[1/9] Generating Before & After Circuit Comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.9))  # IEEE double-column width

    # --- LEFT: Traditional Oracle ---
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 6)
    ax1.set_title("(a) Traditional Exact-Match Oracle", fontsize=10, fontweight='bold')
    ax1.axis('off')

    # Qubit lines
    labels = ['$|node_0\\rangle$', '$|node_1\\rangle$', '$|score\\rangle$', '$|phase\\rangle$']
    y_positions = [5, 4, 2, 0.8]
    for y, lbl in zip(y_positions, labels):
        ax1.plot([1.5, 9], [y, y], 'k-', linewidth=0.8, zorder=0.5)
        ax1.text(0.1, y, lbl, fontsize=8, va='center', fontfamily='serif')

    # Hadamard blocks
    for y in [5, 4]:
        ax1.add_patch(FancyBboxPatch((1.8, y-0.25), 0.6, 0.5, boxstyle="round,pad=0.05", fc='#B3E5FC', ec='black', lw=1))
        ax1.text(2.1, y, 'H', fontsize=9, ha='center', va='center', fontweight='bold')

    # Collision checker
    ax1.add_patch(FancyBboxPatch((3.0, 0.5), 1.2, 5.0, boxstyle="round,pad=0.05", fc='#C8E6C9', ec='#2E7D32', lw=1.5))
    ax1.text(3.6, 3.0, 'Edge\nCheck', fontsize=8, ha='center', va='center', fontweight='bold')

    # Adder block
    ax1.add_patch(FancyBboxPatch((4.6, 0.5), 1.0, 5.0, boxstyle="round,pad=0.05", fc='#FFF9C4', ec='#F57F17', lw=1.5))
    ax1.text(5.1, 3.0, 'Adder', fontsize=8, ha='center', va='center', fontweight='bold')

    # MASSIVE MCX block (the problem)
    ax1.add_patch(FancyBboxPatch((6.0, 0.5), 1.8, 5.0, boxstyle="round,pad=0.05", fc='#FFCDD2', ec='#C62828', lw=2.5))
    ax1.text(6.9, 3.5, '$S$-qubit\nMCX', fontsize=9, ha='center', va='center', fontweight='bold', color='#C62828')
    ax1.text(6.9, 2.2, '$\\mathcal{O}(S^2)$\ndepth', fontsize=8, ha='center', va='center', color='#C62828', style='italic')

    # Uncompute
    ax1.add_patch(FancyBboxPatch((7.95, 0.5), 1.95, 5.0, boxstyle="round,pad=0.05", fc='#E0E0E0', ec='gray', lw=1))
    ax1.text(8.925, 3.0, 'Uncompute', fontsize=8, ha='center', va='center')

    # --- RIGHT: Proposed Oracle ---
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 6)
    ax2.set_title("(b) Proposed MSB-Overflow Oracle", fontsize=10, fontweight='bold')
    ax2.axis('off')

    labels2 = ['$|node_0\\rangle$', '$|node_1\\rangle$', '$|score\\rangle$', '$|msb\\rangle$', '$|phase\\rangle$']
    y_pos2 = [5, 4, 2.5, 1.3, 0.5]
    for y, lbl in zip(y_pos2, labels2):
        ax2.plot([1.8, 9], [y, y], 'k-', linewidth=0.8, zorder=0.5)
        ax2.text(0.1, y, lbl, fontsize=8, va='center', fontfamily='serif')

    # Custom State Prep
    for y in [5, 4]:
        ax2.add_patch(FancyBboxPatch((2.0, y-0.25), 0.8, 0.5, boxstyle="round,pad=0.05", fc='#E1BEE7', ec='#6A1B9A', lw=1))
        ax2.text(2.4, y, '$\\mathcal{A}$', fontsize=9, ha='center', va='center', fontweight='bold')

    # Offset init (X gates on score)
    ax2.add_patch(FancyBboxPatch((3.2, 2.2), 0.7, 0.6, boxstyle="round,pad=0.05", fc='#BBDEFB', ec='#1565C0', lw=1))
    ax2.text(3.55, 2.5, 'Offset\n$X$', fontsize=8, ha='center', va='center', fontweight='bold')

    # Edge Check
    ax2.add_patch(FancyBboxPatch((4.2, 0.2), 1.0, 5.3, boxstyle="round,pad=0.05", fc='#C8E6C9', ec='#2E7D32', lw=1.5))
    ax2.text(4.7, 3.0, 'Edge\nCheck', fontsize=8, ha='center', va='center', fontweight='bold')

    # Adder
    ax2.add_patch(FancyBboxPatch((5.5, 0.2), 0.8, 5.3, boxstyle="round,pad=0.05", fc='#FFF9C4', ec='#F57F17', lw=1.5))
    ax2.text(5.9, 3.0, 'Adder', fontsize=8, ha='center', va='center', fontweight='bold')

    # SINGLE CNOT (the solution!)
    ax2.plot(7.0, 1.3, 'o', markersize=10, color='#1B5E20', markerfacecolor='#1B5E20')  # control dot
    ax2.plot([7.0, 7.0], [1.3, 0.5], '-', color='#1B5E20', lw=2)  # vertical line
    ax2.plot(7.0, 0.5, 'o', markersize=14, color='#1B5E20', markerfacecolor='white', markeredgewidth=2)  # target
    ax2.plot([6.86, 7.14], [0.5, 0.5], '-', color='#1B5E20', lw=2)  # plus horizontal
    ax2.plot([7.0, 7.0], [0.36, 0.64], '-', color='#1B5E20', lw=2)  # plus vertical

    ax2.annotate('Single CNOT\n$\\mathcal{O}(1)$', xy=(7.0, 0.9), xytext=(7.0, 3.5), ha='center',
                fontsize=8, fontweight='bold', color='#1B5E20',
                arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec='#1B5E20'))

    # Uncompute
    ax2.add_patch(FancyBboxPatch((7.7, 0.2), 2.0, 5.3, boxstyle="round,pad=0.05", fc='#E0E0E0', ec='gray', lw=1))
    ax2.text(8.7, 3.0, 'Uncompute', fontsize=8, ha='center', va='center')

    plt.tight_layout()
    plt.savefig('fig_before_after.pdf')
    plt.close()
    print("   -> fig_before_after.pdf saved.")


# ============================================================
# FIGURE 2: Space-Time Pareto Frontier
# ============================================================
def generate_fig2_pareto():
    print("[2/9] Generating Space-Time Pareto Frontier...")
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    # Data for Frustrated K4 (from Table 2)
    data = [
        ('Min Width',  14, 2217, 's', '#1565C0'),
        ('Min Depth',  20, 385,  '^', '#C62828'),
        ('Balanced',   17, 1289, '*', '#2E7D32'),
    ]

    for name, q, d, marker, color in data:
        ms = 14 if name == 'Balanced' else 9
        zorder = 10 if name == 'Balanced' else 5
        ax.scatter(q, d, s=ms**2, marker=marker, c=color,
                  edgecolors='black', linewidth=0.8, zorder=zorder, label=name)

    # Pareto frontier line
    pts = sorted([(14, 2217), (17, 1289), (20, 385)])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], '--', color='gray', alpha=0.4, lw=1)

    # Annotations — positioned to avoid all overlaps
    ax.annotate('Min Width\n(fewest qubits)', xy=(14, 2217),
               xytext=(16.5, 2350), fontsize=8, ha='center',
               arrowprops=dict(arrowstyle='->', lw=0.8, color='#1565C0'),
               color='#1565C0', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#1565C0', alpha=0.8))

    ax.annotate('Min Depth\n(shallowest)', xy=(20, 385),
               xytext=(17.8, 150), fontsize=8, ha='center',
               arrowprops=dict(arrowstyle='->', lw=0.8, color='#C62828'),
               color='#C62828', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#C62828', alpha=0.8))

    ax.annotate('Balanced\n(sweet spot)', xy=(17, 1289),
               xytext=(13.5, 1050), fontsize=8, ha='center',
               arrowprops=dict(arrowstyle='->', lw=1.0, color='#2E7D32'),
               color='#2E7D32', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', fc='#E8F5E9', ec='#2E7D32', alpha=0.9))

    ax.set_xlabel('Qubit Width', fontweight='bold')
    ax.set_ylabel('Transpiled Circuit Depth', fontweight='bold')
    ax.set_title('Frustrated $K_4$: Space-Time Trade-off', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(12.5, 22)
    ax.set_ylim(-50, 2600)

    plt.tight_layout()
    plt.savefig('fig_pareto.pdf')
    plt.close()
    print("   -> fig_pareto.pdf saved.")


# ============================================================
# FIGURE 3: Measurement Probability Histogram
# ============================================================
def generate_fig3_measurement():
    print("[3/9] Generating Measurement Probability Histogram...")

    if HAS_QISKIT:
        # Run real Grover simulation on Frustrated K4 (weights match qaoa_depth_sweep.py)
        N, K = 4, 3
        edges = [(0, 1, 4), (0, 2, 5), (0, 3, 3), (1, 2, 6), (1, 3, 2), (2, 3, 4)]
        target_score = 22
        chunk = math.ceil((1 + math.sqrt(8 * len(edges) - 7)) / 2)
        grover_iters = 2

        print("   Running Grover simulation (Frustrated K4, 2 iterations)...")
        qc = create_weighted_balanced_oracle(N, K, edges, chunk, target_score, grover_iters)
        sim = AerSimulator()
        qc_t = transpile(qc, sim, optimization_level=3)
        result = sim.run(qc_t, shots=8192).result()
        counts = result.get_counts()

        # Sort by count descending
        sorted_states = sorted(counts.items(), key=lambda x: -x[1])
        top_n = 25
        states = [s for s, _ in sorted_states[:top_n]]
        probs = [c / 8192 for _, c in sorted_states[:top_n]]
    else:
        # Mock data
        np.random.seed(42)
        states = [f"{i:08b}" for i in range(25)]
        probs = np.random.uniform(0.005, 0.02, 25)
        # Inject target spikes at indices 0-5
        for i in range(6):
            probs[i] = 0.12
        probs = probs / probs.sum()
        states[0:6] = ['01020120', '02010210', '10020120', '20010210', '01020210', '02010120']

    fig, ax = plt.subplots(figsize=(6.9, 2.8))

    # Determine optimal states (top probability cluster)
    threshold = max(probs) * 0.5
    colors = ['#1B5E20' if p >= threshold else '#BBDEFB' for p in probs]
    edge_colors = ['#1B5E20' if p >= threshold else '#64B5F6' for p in probs]

    bars = ax.bar(range(len(states)), probs, color=colors, edgecolor=edge_colors, linewidth=0.5)

    # Uniform baseline
    K_N = 3**4
    ax.axhline(y=1/K_N, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Uniform $1/K^N = 1/{K_N}$')

    ax.set_xlabel('Measurement Outcome (Top 25 States)', fontweight='bold')
    ax.set_ylabel('Probability', fontweight='bold')
    ax.set_title('Grover Measurement Distribution: Frustrated $K_4$ ($K\\!=\\!3$, 2 iterations)', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, rotation=75, fontsize=8, fontfamily='monospace')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#1B5E20', label='Target optimal states'),
        mpatches.Patch(facecolor='#BBDEFB', edgecolor='#64B5F6', label='Non-optimal states'),
        Line2D([0], [0], color='red', linestyle='--', label=f'Uniform baseline ($1/{K_N}$)')
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_measurement.pdf')
    plt.close()
    print("   -> fig_measurement.pdf saved.")


# ============================================================
# FIGURE 4: IBM Heavy-Hex Mapping
# ============================================================
def generate_fig4_heavy_hex():
    print("[4/9] Generating IBM Heavy-Hex Mapping...")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Build a simplified Heavy-Hex lattice (3x3 section)
    G = nx.Graph()
    # Heavy-hex: hexagonal grid with extra "bridge" qubits
    # Row 0 (data qubits)
    row0 = [(0, 2), (2, 2), (4, 2), (6, 2)]
    # Row 1 (bridge qubits between rows)
    row_bridge_01 = [(1, 1.5), (3, 1.5), (5, 1.5)]
    # Row 1 data
    row1 = [(0, 1), (2, 1), (4, 1), (6, 1)]
    # Row 2 bridge
    row_bridge_12 = [(1, 0.5), (3, 0.5), (5, 0.5)]
    # Row 2 data
    row2 = [(0, 0), (2, 0), (4, 0), (6, 0)]

    all_nodes = row0 + row_bridge_01 + row1 + row_bridge_12 + row2
    pos = {}
    for i, (x, y) in enumerate(all_nodes):
        G.add_node(i)
        pos[i] = (x, y)

    # Edges (nearest-neighbor in heavy-hex)
    hex_edges = [
        (0,1),(1,2),(2,3),  # row0 horizontal
        (0,4),(2,5),(4,6),  # row0-bridge
        (4,7),(5,8),(6,9),  # bridge-row1
        (7,8),(8,9),(9,10), # row1 horizontal
        (7,11),(9,12),(10,13), # row1-bridge
        (11,14),(12,15),(13,16), # bridge-row2
        (14,15),(15,16),(16,17), # row2 horizontal
    ]
    G.add_edges_from(hex_edges)

    # Draw the lattice
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#BDBDBD', width=1.5, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=200, node_color='#ECEFF1', edgecolors='#78909C', linewidths=1)

    # Overlay S4 mapping: node 0 (hub) -> qubit 8, nodes 1,2,3 -> qubits 5, 9, 12
    s4_mapping = {8: '$v_0$ (hub)', 5: '$v_1$', 9: '$v_2$', 12: '$v_3$'}
    s4_qubits = list(s4_mapping.keys())
    s4_colors = ['#E53935' if q == 8 else '#1E88E5' for q in s4_qubits]

    nx.draw_networkx_nodes(G, pos, nodelist=s4_qubits, ax=ax,
                          node_size=350, node_color=s4_colors, edgecolors='black', linewidths=1.5)

    for q, label in s4_mapping.items():
        offset = (0.3, 0.25) if q != 12 else (0.3, -0.25)
        ax.annotate(label, xy=pos[q], xytext=(pos[q][0]+offset[0], pos[q][1]+offset[1]),
                   fontsize=8, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', lw=0.8),
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.9))

    # Draw SWAP paths
    swap_paths = [(5, 8), (8, 9), (9, 12)]
    nx.draw_networkx_edges(G, pos, edgelist=swap_paths, ax=ax,
                          edge_color='#E53935', width=2.5, style='dashed', alpha=0.8)

    ax.set_title('Heavy Star $S_4$ Mapped onto\nIBM Heavy-Hex Lattice', fontsize=10, fontweight='bold')
    ax.axis('off')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E53935', markersize=10, markeredgecolor='black', label='Hub node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E88E5', markersize=10, markeredgecolor='black', label='Leaf nodes'),
        Line2D([0], [0], color='#E53935', linestyle='--', lw=2, label='SWAP routes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ECEFF1', markersize=8, markeredgecolor='#78909C', label='Idle qubits'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('fig_heavy_hex.pdf')
    plt.close()
    print("   -> fig_heavy_hex.pdf saved.")


# ============================================================
# FIGURE 5: Asymptotic Scaling
# ============================================================
def generate_fig5_scaling():
    print("[5/9] Generating Asymptotic Scaling Line Graph...")
    fig, ax = plt.subplots(figsize=(3.36, 2.9))  # IEEE single-column width

    N_values = [5, 10, 20, 50, 100]
    K = 3
    q = 2  # ceil(log2(3))

    def estimate_edges(N, density='moderate'):
        if density == 'sparse': return int(N * 1.5)
        elif density == 'moderate': return int(N * 2.4)
        else: return int(N * (N-1) / 4)

    # T-depth estimates
    t_depth_mw = []
    t_depth_md = []
    t_depth_bal = []
    t_depth_traditional = []

    for N in N_values:
        E = estimate_edges(N, 'moderate')
        S = math.ceil(math.log2(E * 10 + 1)) + 1  # ~max weight per edge = 10
        D_adder = 4 * S  # Toffoli gates per adder * 7 T-gates each

        # Min Width: O(E * D_adder)
        t_depth_mw.append(E * D_adder * 7)
        # Min Depth: O(D_adder)
        t_depth_md.append(D_adder * 7)
        # Balanced: O(ceil(E/C) * D_adder)
        C = math.ceil((1 + math.sqrt(8 * E - 7)) / 2)
        t_depth_bal.append(math.ceil(E / C) * D_adder * 7)

    ax.semilogy(N_values, t_depth_mw, 's-', color='#1565C0', label='Min Width', markersize=5, lw=1.5)
    ax.semilogy(N_values, t_depth_md, '^-', color='#C62828', label='Min Depth', markersize=5, lw=1.5)
    ax.semilogy(N_values, t_depth_bal, '*-', color='#2E7D32', label='Balanced', markersize=7, lw=1.5)

    ax.set_xlabel('Number of Nodes ($N$)', fontweight='bold')
    ax.set_ylabel('Estimated $T$-Gate Depth (log scale)', fontweight='bold')
    ax.set_title('Asymptotic $T$-Depth Scaling', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(N_values)

    plt.tight_layout()
    plt.savefig('fig_scaling.pdf')
    plt.close()
    print("   -> fig_scaling.pdf saved.")


# ============================================================
# CIRCUIT DRAWING HELPERS (Pure Matplotlib — no Qiskit draw)
# ============================================================
def _draw_gate(ax, x, y, label, color='#BBDEFB', w=0.5, h=0.5, fontsize=9, ec='black'):
    """Draw a gate box at position (x,y)."""
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.04", fc=color, ec=ec, lw=1))
    ax.text(x, y, label, fontsize=fontsize, ha='center', va='center', fontweight='bold')

def _draw_cnot(ax, ctrl_y, tgt_y, x, color='black'):
    """Draw a CNOT gate: control dot + target circle."""
    ax.plot(x, ctrl_y, 'o', markersize=6, color=color, markerfacecolor=color, zorder=5)
    ax.plot([x, x], [ctrl_y, tgt_y], '-', color=color, lw=1.2, zorder=4)
    ax.plot(x, tgt_y, 'o', markersize=12, color=color, markerfacecolor='white', markeredgewidth=1.5, zorder=5)
    ax.plot([x-0.08, x+0.08], [tgt_y, tgt_y], '-', color=color, lw=1.5, zorder=6)
    ax.plot([x, x], [tgt_y-0.08, tgt_y+0.08], '-', color=color, lw=1.5, zorder=6)

def _draw_toffoli(ax, ctrl_ys, tgt_y, x, color='black'):
    """Draw a multi-controlled X (MCX/Toffoli)."""
    for cy in ctrl_ys:
        ax.plot(x, cy, 'o', markersize=6, color=color, markerfacecolor=color, zorder=5)
    all_ys = ctrl_ys + [tgt_y]
    ax.plot([x, x], [min(all_ys), max(all_ys)], '-', color=color, lw=1.2, zorder=4)
    ax.plot(x, tgt_y, 'o', markersize=12, color=color, markerfacecolor='white', markeredgewidth=1.5, zorder=5)
    ax.plot([x-0.08, x+0.08], [tgt_y, tgt_y], '-', color=color, lw=1.5, zorder=6)
    ax.plot([x, x], [tgt_y-0.08, tgt_y+0.08], '-', color=color, lw=1.5, zorder=6)

def _draw_barrier(ax, x, y_min, y_max, label=None):
    """Draw a dashed barrier line."""
    ax.plot([x, x], [y_min - 0.15, y_max + 0.15], '--', color='gray', lw=0.8, alpha=0.6)
    if label:
        ax.text(x, y_max + 0.3, label, fontsize=8, ha='center', color='gray', style='italic')

# ============================================================
# CIRCUIT WALKTHROUGH FIGURES (Pure Matplotlib)
# ============================================================
def generate_circuit_walkthrough_figs():
    # --- Fig 6: State Preparation for K=3 on 2 qubits ---
    print("[6/9] Generating State Prep circuit diagram...")
    fig, ax = plt.subplots(figsize=(3.36, 1.7))  # IEEE single-column width
    ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.5, 1.8)
    ax.axis('off')

    qubits = {1.2: '$q_0$', 0.4: '$q_1$'}
    for y, lbl in qubits.items():
        ax.plot([-0.1, 6.0], [y, y], 'k-', lw=0.8, zorder=0.5)
        ax.text(-0.45, y, lbl, fontsize=9, va='center', fontfamily='serif')
    ax.text(3.0, 1.65, 'Custom State Prep: $K=3$, $q=2$', fontsize=10, ha='center', fontweight='bold')

    # X gate on q1
    _draw_gate(ax, 0.5, 0.4, 'X', color='#FFCDD2', fontsize=9)
    # Ry on q1
    _draw_gate(ax, 1.5, 0.4, '$R_y$', color='#E1BEE7', w=0.6, fontsize=8)
    # X on q1
    _draw_gate(ax, 2.5, 0.4, 'X', color='#FFCDD2', fontsize=9)
    # Controlled-H: q1 controls q0
    ax.plot(3.5, 0.4, 'o', markersize=6, color='black', markerfacecolor='black', zorder=5)
    ax.plot([3.5, 3.5], [0.4, 1.2], '-', color='black', lw=1.2, zorder=4)
    _draw_gate(ax, 3.5, 1.2, 'H', color='#B3E5FC', fontsize=9)
    # X on q1
    _draw_gate(ax, 4.5, 0.4, 'X', color='#FFCDD2', fontsize=9)

    # Output annotation
    ax.annotate('$\\frac{1}{\\sqrt{3}}(|00\\rangle+|01\\rangle+|10\\rangle)$',
               xy=(5.5, 0.8), fontsize=8, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec='#2E7D32', alpha=0.9))

    plt.tight_layout()
    plt.savefig('fig_circuit_state_prep.pdf')
    plt.close()
    print("   -> fig_circuit_state_prep.pdf saved.")

    # --- Fig 7: Edge Collision Checker ---
    print("[7/9] Generating Edge Collision Checker circuit...")
    fig, ax = plt.subplots(figsize=(6.9, 3.0))
    ax.set_xlim(-0.8, 9.5); ax.set_ylim(-0.5, 3.2)
    ax.axis('off')

    wires = {2.6: '$u_0$', 1.8: '$u_1$', 1.0: '$v_0$', 0.2: '$v_1$', -0.2: '', 2.99: ''}
    wire_ys = [2.6, 1.8, 1.0, 0.2]
    ans_y = -0.2
    for y in wire_ys:
        ax.plot([-0.3, 9.0], [y, y], 'k-', lw=0.8, zorder=0.5)
    ax.plot([-0.3, 9.0], [ans_y, ans_y], 'k-', lw=0.8, zorder=0.5)
    labels_y = {'$u_0$': 2.6, '$u_1$': 1.8, '$v_0$': 1.0, '$v_1$': 0.2, '$ans$': ans_y}
    for lbl, y in labels_y.items():
        ax.text(-0.75, y, lbl, fontsize=9, va='center', fontfamily='serif')
    ax.text(4.0, 3.1, 'Edge Collision Evaluation: $C(u) \\neq C(v)$', fontsize=10, ha='center', fontweight='bold')

    # Step 1: CX u0->v0
    _draw_cnot(ax, 2.6, 1.0, 0.5)
    # Step 2: X on v0
    _draw_gate(ax, 1.2, 1.0, 'X', color='#FFCDD2', fontsize=9)
    # Step 3: CX u1->v1
    _draw_cnot(ax, 1.8, 0.2, 2.0)
    # Step 4: X on v1
    _draw_gate(ax, 2.7, 0.2, 'X', color='#FFCDD2', fontsize=9)
    # Step 5: Toffoli v0,v1 -> ans
    _draw_toffoli(ax, [1.0, 0.2], ans_y, 3.5, color='#1565C0')
    # Barrier
    _draw_barrier(ax, 4.3, ans_y, 2.6, 'uncompute')
    # Step 6: X on v1 (uncompute)
    _draw_gate(ax, 5.0, 0.2, 'X', color='#FFCDD2', fontsize=9)
    # Step 7: CX u1->v1 (uncompute)
    _draw_cnot(ax, 1.8, 0.2, 5.7)
    # Step 8: X on v0 (uncompute)
    _draw_gate(ax, 6.4, 1.0, 'X', color='#FFCDD2', fontsize=9)
    # Step 9: CX u0->v0 (uncompute)
    _draw_cnot(ax, 2.6, 1.0, 7.1)
    # Step 10: X on ans (flip logic)
    _draw_gate(ax, 8.0, ans_y, 'X', color='#C8E6C9', fontsize=9, ec='#2E7D32')

    ax.annotate('$|ans\\rangle = |1\\rangle$ iff\n$C(u) \\neq C(v)$', xy=(8.5, -0.2),
               xytext=(8.5, 0.7), fontsize=8, ha='center', fontweight='bold', color='#2E7D32',
               arrowprops=dict(arrowstyle='->', color='#2E7D32'),
               bbox=dict(boxstyle='round,pad=0.2', fc='#E8F5E9', ec='#2E7D32'))

    plt.tight_layout()
    plt.savefig('fig_circuit_edge_check.pdf')
    plt.close()
    print("   -> fig_circuit_edge_check.pdf saved.")

    # --- Fig 8: Carry-Ripple Adder (adding w=3) ---
    print("[8/9] Generating Carry-Ripple Adder circuit...")
    fig, ax = plt.subplots(figsize=(6.9, 3.4))
    ax.set_xlim(-0.8, 9.0); ax.set_ylim(-0.8, 3.5)
    ax.axis('off')

    adder_wires = {2.8: '$ans$', 2.0: '$s_0$', 1.2: '$s_1$', 0.4: '$s_2$', -0.4: '$s_3$ (MSB)'}
    for y, lbl in adder_wires.items():
        ax.plot([-0.3, 8.5], [y, y], 'k-', lw=0.8, zorder=0.5)
        ax.text(-0.75, y, lbl, fontsize=8, va='center', fontfamily='serif')
    ax.text(4.0, 3.3, 'Controlled Add $w=3$ ($11_2$): Bit 0 + Bit 1', fontsize=10, ha='center', fontweight='bold')

    # Incrementor for bit 0 (add 1 starting from s0): ans controls cascade on s0,s1,s2,s3
    ax.add_patch(FancyBboxPatch((0.5, -0.6), 3.0, 3.6, boxstyle="round,pad=0.08",
                 fc='#FFF9C4', ec='#F57F17', lw=1.5, alpha=0.3))
    ax.text(2.0, 3.05, 'Inc from $s_0$', fontsize=10, ha='center', color='#F57F17', fontweight='bold')

    # MCX: ans,s0,s1,s2 -> s3
    _draw_toffoli(ax, [2.8, 2.0, 1.2, 0.4], -0.4, 1.0, color='#E65100')
    # MCX: ans,s0,s1 -> s2
    _draw_toffoli(ax, [2.8, 2.0, 1.2], 0.4, 1.8, color='#E65100')
    # Toffoli: ans,s0 -> s1
    _draw_toffoli(ax, [2.8, 2.0], 1.2, 2.6, color='#E65100')
    # CNOT: ans -> s0
    _draw_cnot(ax, 2.8, 2.0, 3.2)

    # Incrementor for bit 1 (add 1 starting from s1): ans controls cascade on s1,s2,s3
    ax.add_patch(FancyBboxPatch((4.5, -0.6), 2.8, 3.6, boxstyle="round,pad=0.08",
                 fc='#E3F2FD', ec='#1565C0', lw=1.5, alpha=0.3))
    ax.text(5.9, 3.05, 'Inc from $s_1$', fontsize=10, ha='center', color='#1565C0', fontweight='bold')

    # MCX: ans,s1,s2 -> s3
    _draw_toffoli(ax, [2.8, 1.2, 0.4], -0.4, 5.2, color='#1565C0')
    # Toffoli: ans,s1 -> s2
    _draw_toffoli(ax, [2.8, 1.2], 0.4, 6.0, color='#1565C0')
    # CNOT: ans -> s1
    _draw_cnot(ax, 2.8, 1.2, 6.8)

    plt.tight_layout()
    plt.savefig('fig_circuit_adder.pdf')
    plt.close()
    print("   -> fig_circuit_adder.pdf saved.")

    # --- Fig 9: Overflow Detection & Phase Kickback ---
    print("[9/9] Generating Overflow Detection circuit...")
    fig, ax = plt.subplots(figsize=(6.9, 3.0))
    ax.set_xlim(-1.0, 8.0); ax.set_ylim(-0.8, 3.0)
    ax.axis('off')

    of_wires = {2.4: '$s_0$', 1.6: '$s_1$', 0.8: '$s_2$', 0.0: '$s_{MSB}$ (flag)', -0.5: '$|-\\rangle$ (phase)'}
    for y, lbl in of_wires.items():
        ax.plot([-0.3, 7.5], [y, y], 'k-', lw=0.8, zorder=0.5)
        ax.text(-1.0, y, lbl, fontsize=8, va='center', fontfamily='serif')
    ax.text(3.5, 2.8, 'MSB Overflow Detection & Phase Kickback', fontsize=10, ha='center', fontweight='bold')

    # Offset init: X on s0, X on s2 (offset = 5 = 101)
    _draw_gate(ax, 0.5, 2.4, 'X', color='#BBDEFB', ec='#1565C0', fontsize=9)
    _draw_gate(ax, 0.5, 0.8, 'X', color='#BBDEFB', ec='#1565C0', fontsize=9)
    ax.annotate('Offset = 5\n($101_2$)', xy=(0.5, 1.6), xytext=(0.5, -0.8),
               fontsize=8, ha='center', color='#1565C0',
               bbox=dict(boxstyle='round,pad=0.2', fc='#E3F2FD', ec='#1565C0'))

    # Barrier: Offset Init
    _draw_barrier(ax, 1.3, -0.5, 2.4, 'Offset Init')

    # Adder blocks placeholder
    ax.add_patch(FancyBboxPatch((1.8, -0.7), 2.0, 3.3, boxstyle="round,pad=0.08",
                 fc='#FFF9C4', ec='#F57F17', lw=1.5, alpha=0.4))
    ax.text(2.8, 1.0, 'Edge Checks\n+\nAdders\n($\\forall e \\in E$)', fontsize=8, ha='center', va='center',
            fontweight='bold', color='#F57F17',
            bbox=dict(boxstyle='round,pad=0.25', fc='#FFF9C4', ec='none'))

    # Barrier
    _draw_barrier(ax, 4.2, -0.5, 2.4)

    # THE KEY: Single CNOT from MSB to phase
    _draw_cnot(ax, 0.0, -0.5, 5.0)
    ax.annotate('Single CNOT\n$\\mathcal{O}(1)$ depth!', xy=(5.0, -0.25),
               xytext=(5.0, 1.7), fontsize=8, ha='center', fontweight='bold', color='#1B5E20',
               arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec='#1B5E20'))

    # Barrier (label omitted; the figure title already names this stage)
    _draw_barrier(ax, 5.8, -0.5, 2.4)

    # Uncompute
    ax.add_patch(FancyBboxPatch((6.2, -0.7), 1.0, 3.3, boxstyle="round,pad=0.08",
                 fc='#E0E0E0', ec='gray', lw=1, alpha=0.5))
    ax.text(6.7, 1.0, 'Un-\ncompute', fontsize=8, ha='center', va='center', color='gray')

    plt.tight_layout()
    plt.savefig('fig_circuit_overflow.pdf')
    plt.close()
    print("   -> fig_circuit_overflow.pdf saved.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("IEEE TQE Publication Figure Generator")
    print("=" * 60)

    generate_fig1_before_after()
    generate_fig2_pareto()
    generate_fig3_measurement()
    generate_fig4_heavy_hex()
    generate_fig5_scaling()
    generate_circuit_walkthrough_figs()

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)
    print("\nGenerated files (vector PDF):")
    print("  1. fig_before_after.pdf    (Figure 1: Traditional vs Proposed)")
    print("  2. fig_pareto.pdf          (Figure 2: Pareto Frontier)")
    print("  3. fig_measurement.pdf     (Figure 3: Measurement Histogram)")
    print("  4. fig_heavy_hex.pdf       (Figure 4: Heavy-Hex Mapping)")
    print("  5. fig_scaling.pdf         (Figure 5: Asymptotic Scaling)")
    print("  6. fig_circuit_state_prep.pdf  (State Preparation)")
    print("  7. fig_circuit_edge_check.pdf  (Edge Collision Checker)")
    print("  8. fig_circuit_adder.pdf       (Carry-Ripple Adder)")
    print("  9. fig_circuit_overflow.pdf    (MSB Overflow + Kickback)")
    print("\nNext: Compile tqe_weighted.tex with pdflatex (run twice for refs).")
