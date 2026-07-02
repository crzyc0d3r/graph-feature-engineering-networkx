"""6 graph feature engineering techniques with NetworkX.

Recreated from the Daily Dose of Data Science newsletter
"[Hands-on] How to Build Your Own AI Company" (Graph ML section),
received 2026-07-01. https://www.dailydoseofds.com

Techniques:
  1-3) In-degree, out-degree, total degree
  4)   Betweenness centrality
  5)   Closeness centrality
  6)   Eigenvector centrality
"""

import networkx as nx
import pandas as pd


def build_dummy_social_graph() -> tuple[pd.DataFrame, pd.DataFrame, nx.DiGraph]:
    """Create a dummy social-network dataset (accounts + followers) and
    convert it to a directed NetworkX graph.

    Edges point from the follower to the account they follow.
    """
    accounts = pd.DataFrame(
        {
            "account_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "username": [
                "alice", "bob", "carol", "dave",
                "erin", "frank", "grace", "heidi",
            ],
        }
    )

    followers = pd.DataFrame(
        {
            # follower_id follows account_id
            "follower_id": [2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 1, 2, 6, 7, 8, 5],
            "account_id":  [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6],
        }
    )

    G = nx.DiGraph()
    G.add_nodes_from(accounts["account_id"])
    G.add_edges_from(zip(followers["follower_id"], followers["account_id"]))
    return accounts, followers, G


def engineer_graph_features(accounts: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """Attach the 6 graph features to the accounts DataFrame."""
    # 1-3) Node degree features
    accounts["in_degree"] = accounts["account_id"].map(lambda x: G.in_degree(x))
    accounts["out_degree"] = accounts["account_id"].map(lambda x: G.out_degree(x))
    accounts["degree"] = accounts["account_id"].map(lambda x: G.degree(x))

    # 4-6) Node centrality features
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

    accounts["betweenness_centrality"] = accounts["account_id"].map(betweenness)
    accounts["closeness_centrality"] = accounts["account_id"].map(closeness)
    accounts["eigenvector_centrality"] = accounts["account_id"].map(eigenvector)
    return accounts


def plot_graph(G: nx.DiGraph, accounts: pd.DataFrame, path: str = "social_graph.png"):
    """Save a visualization of the social graph (optional)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = dict(zip(accounts["account_id"], accounts["username"]))
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(
        G, pos, labels=labels, node_color="#8ecae6",
        node_size=1200, arrowsize=15, font_size=9,
    )
    plt.title("Dummy social network (follower -> followed)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Graph visualization saved to {path}")


if __name__ == "__main__":
    accounts, followers, G = build_dummy_social_graph()
    accounts = engineer_graph_features(accounts, G)
    pd.set_option("display.width", 140)
    print(accounts.round(4).to_string(index=False))
    try:
        plot_graph(G, accounts)
    except ImportError:
        print("matplotlib not installed - skipping plot")
