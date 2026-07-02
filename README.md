# 6 Graph Feature Engineering Techniques (NetworkX)

Creation of the **Graph ML** section of the [Daily Dose of Data Science](https://www.dailydoseofds.com) newsletter issue *"[Hands-on] How to Build Your Own AI Company"* (2026-07-01), by Avi Chawla & Akshay Pachaar. All credit for the original tutorial goes to DDODS; the code in the newsletter was published as screenshots, so it is re-implemented here as runnable Python.

Like images, text, and tabular data, graph datasets have features that can be engineered to boost model performance. This project builds a dummy social graph (`accounts` + `followers` tables converted to a directed `networkx` graph) and computes six must-know features per node:

1. **In-degree** — incoming edges (followers); high values suggest influence.
2. **Out-degree** — outgoing edges (followings); high values suggest an active user.
3. **Total degree** — sum of both.
4. **Betweenness centrality** — how often a node lies on shortest paths between others (bridge nodes).
5. **Closeness centrality** — reciprocal of summed shortest-path distances to all other nodes (efficient information spreaders).
6. **Eigenvector centrality** — influence amplified by being connected to other influential nodes.

## Run

```bash
pip install -r requirements.txt
python graph_features.py
```

Prints the feature-enriched `accounts` DataFrame and saves `social_graph.png`.

## Source

Original article & GNN crash course: https://www.dailydoseofds.com
