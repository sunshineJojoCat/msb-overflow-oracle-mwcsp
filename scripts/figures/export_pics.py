import sys, os
import networkx as nx
import matplotlib.pyplot as plt

# The current script is in notebooks/refactored/TQE_Template
# main.py is in the root (../../../)
root_path = os.path.abspath(os.path.join(os.getcwd(), '../../../'))
sys.path.append(root_path)

from main import make_circuit
from balanced import balanced_system
try:
    from notebooks.refactored.my_oracles import make_circuit_custom_init
except ImportError:
    # also add refactored just in case
    sys.path.append(os.path.join(root_path, 'notebooks', 'refactored'))
    from my_oracles import make_circuit_custom_init

# Create the Graph
G = nx.Graph()
G.add_nodes_from(range(4))
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

# Save the Graph Image
plt.figure(figsize=(4,3))
nx.draw(G, with_labels=True, node_color='lightblue', node_size=800, font_weight='bold')
plt.savefig('figures/sample_graph.pdf', bbox_inches='tight')
plt.close()

# Save the Circuit Image
qc = make_circuit_custom_init(G, 3, balanced_system, {}, grover_iterations=1)
qc.draw('mpl', filename='figures/circuit_diagram.pdf', fold=60)
print("Saved pics.")
