# Research Resources

Self-contained reproducibility package for the paper

> **Advancing Quantum Graph Coloring: Custom State Preparation and MSB-Overflow Oracles for the Maximum Weight Colorable Subgraph Problem**
> W. Issariyawat and K. Suksen (Chulalongkorn University), 2026.

Every table and figure in the paper can be regenerated from a script in this
folder. The source tex/bib are also included so the PDF can be rebuilt from
scratch.

---

## Public access

| Resource | Link |
|---|---|
| Source repository (live) | https://github.com/sunshineJojoCat/msb-overflow-oracle-mwcsp |
| Archived snapshot for the paper | Zenodo [DOI:10.5281/zenodo.20454051](https://doi.org/10.5281/zenodo.20454051) (release `v1.0.1`) |
| License | MIT (see `LICENSE`) |

## Citing the code

If you reuse any of these scripts, please cite the Zenodo snapshot in
addition to the paper:

```bibtex
@software{msb_overflow_oracle_2026,
  author    = {Issariyawat, Witchapat and Suksen, Kamonluk},
  title     = {{MSB-Overflow Oracle for MWCSP: Reproducibility Package}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.1},
  doi       = {10.5281/zenodo.20454051},
  url       = {https://github.com/sunshineJojoCat/msb-overflow-oracle-mwcsp}
}
```

---

## Folder layout

```
research_resources/
├── README.md                              ← this file
├── paper/
│   ├── tqe_weighted.tex                   ← LaTeX source
│   ├── references.bib                     ← BibTeX (22 entries)
│   └── tqe_weighted.pdf                   ← compiled output
├── figures/                               ← 8 published PNGs (one per \includegraphics)
│   ├── fig_all_topologies.png
│   ├── fig_before_after.png
│   ├── fig_circuit_adder.png
│   ├── fig_circuit_edge_check.png
│   ├── fig_circuit_overflow.png
│   ├── fig_circuit_state_prep.png
│   ├── fig_measurement.png
│   └── fig_scaling.png
└── scripts/
    ├── reproducibility/                   ← regenerates the paper's numerical claims
    │   ├── verify_table2.py               ← Table II (and re-uses oracle code in noisy/extend)
    │   ├── noisy_simulation.py            ← Table VI
    │   └── extend_table_scaling.py        ← Table VII
    ├── figures/                           ← regenerates the published PNGs
    │   ├── generate_all_figures.py        ← 8 of 8 paper figures
    │   ├── generate_all_topologies.py     ← topology grid (Fig 2)
    │   ├── generate_publication_figures.py ← legacy variants
    │   └── export_pics.py                 ← misc PNG export helper
    └── baselines/                         ← QAOA comparison (feeds Table V)
        ├── simulate_qaoa_paper_topologies.py
        ├── simulate_qaoa_mwcsp_full.py
        ├── simulate_qaoa_qiskit.py
        ├── qaoa_depth_sweep.py
        ├── test_full_qaoa_transpile.py
        └── test_liu_binary_qaoa.py
```

---

## Setup

Tested with Python 3.9. Create a fresh virtualenv and install from the
pinned dependency list:

```bash
python3.9 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The pinned versions in `requirements.txt` are the exact ones used to
produce every numerical result in the paper (Qiskit 2.2.3, Qiskit-Aer
0.17.2, with NumPy / SciPy / NetworkX / Matplotlib / pandas / pypdf).
Docker is used only for the LaTeX rebuild (texlive:latest); see
"Paper" below.

---

## Reproducing the paper's tables

| Table | Caption | Script | Runtime |
|---|---|---|---|
| **I** (`tab:correctness`) | Quantum vs. classical brute-force | `tqe_experimental_results.py` (sibling notebook) | seconds |
| **II** (`tab:results_n`) | Transpiled space-time metrics, named topologies (N=3..5) | `scripts/reproducibility/verify_table2.py` | ~30 s |
| **III** (`tab:tgate`) | Theoretical Toffoli/T-gate counts | derived analytically from §III formulas — no script | n/a |
| **IV** (`tab:scalability`) | Asymptotic projections for N=20/50/100 | derived from §III + Eq. 17 + surface-code overhead — see `scripts/figures/generate_all_figures.py:generate_fig5_scaling()` for the calculation | seconds |
| **V** (`tab:empirical_scaling`) | QAOA (p=7) vs MSB-Overflow gate counts | `scripts/baselines/test_full_qaoa_transpile.py` + `simulate_qaoa_paper_topologies.py` | minutes |
| **VI** (`tab:noisy`) | Min Width oracle under Heron-class depolarizing noise | `scripts/reproducibility/noisy_simulation.py` | ~5 min |
| **VII** (`tab:scaling_measured`) | Random graphs N=6..12, two seeds each | `scripts/reproducibility/extend_table_scaling.py` | ~5 min |

Run each as:

```bash
cd scripts/reproducibility
../../../.venv/bin/python -u verify_table2.py      # Table II
../../../.venv/bin/python -u noisy_simulation.py   # Table VI (uses verify_table2 oracles)
../../../.venv/bin/python -u extend_table_scaling.py  # Table VII
```

(`noisy_simulation.py` and `extend_table_scaling.py` import the oracle
constructors from `verify_table2.py`, so they must sit in the same
directory — which they already do here.)

---

## Regenerating the figures

| Figure | Caption | Source |
|---|---|---|
| **1** (`fig:circuit_sp`) | Custom state preparation subcircuit (K=3, q=2) | `generate_all_figures.py` → `generate_circuit_walkthrough_figs()` |
| **2** (`fig:circuit_edge`) | Edge collision evaluation (q=2) | same |
| **3** (`fig:circuit_adder`) | Controlled carry-ripple adder for w=3 | same |
| **4** (`fig:circuit_overflow`) | MSB-Overflow detection + phase kickback | same |
| **5** (`fig:before_after`) | Traditional vs MSB-Overflow oracle | `generate_all_figures.py` → `generate_fig1_before_after()` |
| **6** (`fig:topologies`) | All five benchmark graphs | `generate_all_topologies.py` |
| **7** (`fig:measurement`) | Grover measurement histogram for Frustrated K4 | `generate_all_figures.py` → `generate_fig3_measurement()` (real Aer sim) |
| **8** (`fig:scaling`) | Asymptotic T-depth scaling | `generate_all_figures.py` → `generate_fig5_scaling()` |

To regenerate every figure at once:

```bash
cd scripts/figures
../../../.venv/bin/python generate_all_figures.py
../../../.venv/bin/python generate_all_topologies.py
```

PNGs land next to the script; copy them into `../../figures/` (or directly
into `paper/`) to overwrite the published versions.

---

## Rebuilding the paper PDF

`paper/` is hermetic — `IEEEtran.cls` is already included. Just:

```bash
cd paper
docker run --rm -v "$(pwd):/work" -w /work texlive/texlive:latest \
  bash -c "pdflatex -interaction=nonstopmode tqe_weighted.tex && \
           bibtex tqe_weighted && \
           pdflatex -interaction=nonstopmode tqe_weighted.tex && \
           pdflatex -interaction=nonstopmode tqe_weighted.tex"
```

The published copy in `paper/tqe_weighted.pdf` (16 pages) was built with
this exact command.

---

## Oracle code (canonical source)

The three oracle variants — Min Width, Min Depth, Balanced — and the
state-preparation / adder / classical brute-force helpers live in
`scripts/reproducibility/verify_table2.py`. The same constructors are
imported by `noisy_simulation.py` and `extend_table_scaling.py`, so any
algorithmic correction needs to be made in a single place.

Quick reference:

- `create_min_width_oracle(N, K, edges, target, grover_iters=1)`
- `create_min_depth_oracle(N, K, edges, target, grover_iters=1)`
- `create_balanced_oracle(N, K, edges, chunk_size, target, grover_iters=1)`
- `belletti_auto_chunk_size(E)` → optimal $C^*$ from §III
- `classical_best_score(N, K, edges)` → brute-force ground truth

The Min Width register layout is
`coloring(N·q) + edge_anc(1) + counter(S) + phase_anc(1)`,
giving the qubit-count formulas reported in §III and verified in Table II:

- $Q_{MW}\!=\!N\!\cdot\!q + S + 2$
- $Q_{MD}\!=\!N\!\cdot\!q + S + |E| + 1$
- $Q_{Bal}\!=\!N\!\cdot\!q + S + C + 1$

with $S = \lceil\log_2(W_{\max}+1)\rceil + 1$ (including the MSB overflow bit).

---

## Verification commit trail

The reproducibility scripts in this package back the following commits in
the main repository (most recent first):

- `5218260` — extend empirical scaling table to N=6..12 (adds Table VII)
- `19c0088` — trim flowery prose to a technical-neutral register
- `8c7c7d3` — add empirical noise sweep on IBM Heron-class depolarising model (adds Table VI)
- `89af451` — add Related Work on quantum arithmetic and threshold comparators
- `a425009` — add limitations subsection: Grover scaling + Target finding
- `15c249d` — fix Table II numbers and reconcile Q formulas with code

Each commit's diff is the diff that produced the corresponding paper section,
so running the scripts and comparing against the commit messages is a quick
sanity check.
