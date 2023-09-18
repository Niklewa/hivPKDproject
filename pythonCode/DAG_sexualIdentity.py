import networkx as nx
import matplotlib.pyplot as plt

dag = nx.DiGraph()
dag.add_edges_from([('H', 'HIV'), ('H', 'A'),
                    ('A', 'HIV'), ('Pn', 'HIV'),
                    ('H', 'Pn'), ('S', 'HIV'),
                    ('Prot', 'HIV'), ('Pn', 'Prot')])

# Draw the DAG
nx.draw(dag, with_labels=True,
        node_color= '#AEAEAE',
        node_size= 1000)
plt.savefig('DAG_HIV_HOMO.png')
plt.show()

# HIV - being HIV positive
# H - homosequal identity
# A - anal intercourse
# Pn - Sexual Partners number
# Prot - using protection
# S - sex
# U - unobserved
# HRB - High Risk Behavour

dag2 = nx.DiGraph()
dag2.add_edges_from([('H', 'HRB'), ('S', 'HRB'), ('U', 'HRB'), ('HRB', 'HIV')])
pos = nx.spring_layout(dag2, seed=42) 
nx.draw(dag2, pos, with_labels=True, node_color='#AEAEAE', node_size=1000)
plt.show()



plt.savefig('DAG_HIV_HOMOpost.png')


# plt.savefig('dag.png') for saving a DAG as a seperate file


# for quarto
#engine: knitr




# Wrong DAG

dagWrong = nx.DiGraph()
dagWrong.add_edges_from([ ('Male', 'HIV'), ('Homo', 'HIV'), ('U', 'HIV')])
pos = nx.spring_layout(dagWrong, seed=42) 
nx.draw(dagWrong, pos, with_labels=True, node_color='#AEAEAE', node_size=1000)
plt.show()



dagWrong = nx.DiGraph()
dagWrong.add_edges_from([('Male', 'HIV'), ('Homo', 'HIV'), ('U', 'HIV')])

# Define node colors
node_colors = {
    'Male': '#AEAEAE',  
    'Homo': '#AEAEAE',  
    'U': 'blue',        
    'HIV': '#AEAEAE'    
}

# Specify node colors in the draw function
pos = nx.spring_layout(dagWrong, seed=42)
nx.draw(dagWrong, pos, with_labels=True, node_color=[node_colors[node] for node in dagWrong.nodes()], node_size=1000)

plt.show()




import networkx as nx
import matplotlib.pyplot as plt
import pydot

# Create the DAG
dagWrong = nx.DiGraph()
dagWrong.add_edges_from([('Male', 'HIV'), ('Homo', 'HIV'), ('U', 'HIV')])

# Define node colors
node_colors = {
    'Male': '#AEAEAE',  # Gray color for 'Male'
    'Homo': '#AEAEAE',  # Gray color for 'Homo'
    'U': 'blue',        # Blue color for 'U'
    'HIV': '#AEAEAE'    # Gray color for 'HIV'
}

# Create a directed graph for vertical orientation
G = nx.DiGraph()
G.add_edges_from(dagWrong.edges())

# Specify node colors in the graph
node_color = [node_colors.get(node, 'black') for node in G.nodes()]

# Create a vertical layout using pydot
pydot_graph = nx.drawing.nx_pydot.to_pydot(G)
pydot_graph.set_rankdir('TB')  # Set direction to top to bottom

# Plot the graph
plt.figure(figsize=(6, 4))
plt.axis('off')
plt.imshow(pydot_graph.create_png(), aspect='auto')
plt.show()
