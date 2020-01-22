##### [Networkx : Convert multigraph into simple graph with weighted edges](https://stackoverflow.com/questions/15590812/networkx-convert-multigraph-into-simple-graph-with-weighted-edges)

```py
import networkx as nx
# weighted MultiGraph
M = nx.MultiGraph()
M.add_edge(1,2,weight=7)
M.add_edge(1,2,weight=19)
M.add_edge(2,3,weight=42)

# create weighted graph from M
G = nx.Graph()
for u,v,data in M.edges(data=True):
    w = data['weight'] if 'weight' in data else 1.0
    if G.has_edge(u,v):
        G[u][v]['weight'] += w
    else:
        G.add_edge(u, v, weight=w)

print(G.edges(data=True))
# [(1, 2, {'weight': 26}), (2, 3, {'weight': 42})]
```



##### [Easiest way to draw a full neo4j graph in networkx](https://stackoverflow.com/questions/33535018/easiest-way-to-draw-a-full-neo4j-graph-in-networkx)

Use ipython-cypher to write a Cypher query and then convert the results to a NetworkX graph. Install it with `pip install ipython-cypher`.

```py
import networkx as nx
%load_ext cypher
%matplotlib inline

results = %cypher MATCH p = ()-[]-() RETURN p

g = results.get_graph()

nx.draw(g)
```

Drawing your entire graph is expensive if it is large. Consider only drawing the subgraphs you're interested in. You'll also have to tweak the query slightly if you want nodes with degree 0.

more: <https://www.quora.com/How-do-I-convert-a-Neo4j-graph-to-a-Networkx-graph>

#### plot

##### [Plotting networkx graph with node labels defaulting to node name](https://stackoverflow.com/questions/28533111/plotting-networkx-graph-with-node-labels-defaulting-to-node-name)

 just add `with_labels=True` to the `nx.draw` call.

```python
import networkx as nx
import pylab as plt

G=nx.Graph()
# Add nodes and edges
G.add_edge("Node1", "Node2")
nx.draw(G, with_labels = True)


labeldict = {}
labeldict["Node1"] = "shopkeeper"
labeldict["Node2"] = "angry man with parrot"

nx.draw(G, labels=labeldict, with_labels = True)
```



```python
import networkx as nx
import community

G = nx.random_graphs.powerlaw_cluster_graph(300, 1, .4)

part = community.best_partition(G)
values = [part.get(node) for node in G.nodes()]

nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=False)
mod = community.modularity(part,G)
print("modularity:", mod)
```





##### [How do I customize the display of edge labels in networkx?](https://stackoverflow.com/questions/34617307/how-do-i-customize-the-display-of-edge-labels-in-networkx)

```
nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
```



