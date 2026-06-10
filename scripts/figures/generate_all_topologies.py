import matplotlib.pyplot as plt
import networkx as nx

# Rendered at the true double-column on-page width (~6.9 in) and exported as a
# vector PDF so \includegraphics does not down-scale the figure (which had
# previously shrunk all labels to ~half their nominal size). See the companion
# note in generate_all_figures.py.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight'
})

fig, axes = plt.subplots(1, 5, figsize=(6.9, 2.2))
node_color = '#A0CBE2'

# 1. Triangle
G1 = nx.cycle_graph(3)
edges1 = list(G1.edges())
edge_weights1 = {edges1[0]: 1, edges1[1]: 2, edges1[2]: 3}
nx.set_edge_attributes(G1, edge_weights1, 'weight')
pos1 = nx.spring_layout(G1, seed=42)
nx.draw(G1, pos1, ax=axes[0], with_labels=True, node_color=node_color, node_size=150, font_weight='bold', font_size=9)
nx.draw_networkx_edge_labels(G1, pos1, edge_labels=edge_weights1, ax=axes[0], font_color='red', font_weight='bold', font_size=8)
axes[0].set_title('Weighted Triangle\n(N=3, E=3)')

# 2. Heavy Star S4
G2 = nx.star_graph(3)
edge_weights2 = {(0, 1): 4, (0, 2): 2, (0, 3): 5}
nx.set_edge_attributes(G2, edge_weights2, 'weight')
pos2 = nx.spring_layout(G2, seed=42)
nx.draw(G2, pos2, ax=axes[1], with_labels=True, node_color=node_color, node_size=150, font_weight='bold', font_size=9)
nx.draw_networkx_edge_labels(G2, pos2, edge_labels=edge_weights2, ax=axes[1], font_color='red', font_weight='bold', font_size=8)
axes[1].set_title('Heavy Star $S_4$\n(N=4, E=3)')

# 3. Diagonal Square
G3 = nx.cycle_graph(4)
G3.add_edge(0, 2, weight=10)
edge_weights3 = {(0, 1): 2, (1, 2): 2, (2, 3): 2, (3, 0): 2, (0, 2): 10} # Perimeter weights 2 to make optimal score = 18
pos3 = nx.circular_layout(G3)
nx.draw(G3, pos3, ax=axes[2], with_labels=True, node_color=node_color, node_size=150, font_weight='bold', font_size=9)
nx.draw_networkx_edge_labels(G3, pos3, edge_labels=edge_weights3, ax=axes[2], font_color='red', font_weight='bold', font_size=8)
axes[2].set_title('Diagonal Square\n(N=4, E=5)')

# 4. Path P5
G4 = nx.path_graph(5)
edges4 = list(G4.edges())
edge_weights4 = {edges4[0]: 3, edges4[1]: 7, edges4[2]: 2, edges4[3]: 6}
nx.set_edge_attributes(G4, edge_weights4, 'weight')
pos4 = {i: (i, i) for i in range(5)}  # uniform diagonal: equal spacing, every edge visible
nx.draw(G4, pos4, ax=axes[3], with_labels=True, node_color=node_color, node_size=150, font_weight='bold', font_size=9)
nx.draw_networkx_edge_labels(G4, pos4, edge_labels=edge_weights4, ax=axes[3], font_color='red', font_weight='bold', font_size=8)
axes[3].set_title('Path $P_5$\n(N=5, E=4)')

# 5. Frustrated K4
G5 = nx.complete_graph(4)
# 5 edges with weight 10, 1 edge with weight 2 to force a specific optimal collision (score=50, 6 optimal states)
edges5 = list(G5.edges())
edge_weights5 = {e: 10 for e in edges5}
edge_weights5[edges5[0]] = 2 # Make one edge have weight 2
pos5 = nx.circular_layout(G5)
nx.draw(G5, pos5, ax=axes[4], with_labels=True, node_color=node_color, node_size=150, font_weight='bold', font_size=9)
nx.draw_networkx_edge_labels(G5, pos5, edge_labels=edge_weights5, ax=axes[4], font_color='red', font_weight='bold', font_size=8, label_pos=0.3)
axes[4].set_title('Frustrated $K_4$\n(N=4, E=6)')

plt.tight_layout()
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'fig_all_topologies.pdf')
plt.savefig(output_path)
print(f"Successfully generated {output_path}")
