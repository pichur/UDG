import itertools
import argparse
import networkx as nx
import igraph as ig
from networkx.readwrite.graph6 import to_graph6_bytes

"""Convert between custom edge-list format and canonical graph6."""

def edge_list_to_graph(edge_list: str) -> nx.Graph:
    """Parse edge-list formatted as 'n: u_1,v_1 ; u_2,v_2 ...'."""
    edge_list = edge_list.strip()
    if not edge_list:
        raise ValueError("Empty graph description")
    parts = edge_list.split(":", 1)
    if len(parts) != 2:
        raise ValueError("Missing ':' separator")
    n = int(parts[0].strip())
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges_part = parts[1]
    if edges_part.strip() == "":
        return G
    # remove spaces around semicolons and commas
    edges = [e.strip() for e in edges_part.split(";") if e.strip()]
    for e in edges:
        u_str, v_str = [x.strip() for x in e.split(",")]
        u = int(u_str) - 1
        v = int(v_str) - 1
        if u < 0 or v < 0 or u >= n or v >= n:
            raise ValueError("Vertex index out of range")
        if u != v:
            G.add_edge(u, v)
    return G

def g6_to_graph(g6: str) -> nx.Graph:
    """Parse graph6 formatted string."""
    g6 = g6.strip()
    if not g6:
        raise ValueError("Empty graph description")
    try:
        G = nx.from_graph6_bytes(g6.encode())
    except Exception as e:
        raise ValueError(f"Invalid graph6 format: {e}")
    return G

def graph_to_edge_list(G: nx.Graph) -> str:
    n = len(G)
    edges = sorted({tuple(sorted((u + 1, v + 1))) for u, v in G.edges()})
    edges_str = " ; ".join(f"{u},{v}" for u, v in edges)
    return f"{n}: {edges_str}" if edges_str else f"{n}:"

def graph_to_g6(G: nx.Graph) -> str:
    g6 = to_graph6_bytes(G).decode().strip()
    return g6

def canonize_graph(G: nx.Graph) -> nx.Graph:
    """Return a canonical form of the graph using a sorted adjacency labeling."""
    g_ig = ig.Graph.from_networkx(G)
    # zwraca permutację: indeks -> nowa_etykieta
    perm = g_ig.canonical_permutation(color=None)   # uwzględnisz 'color', jeżeli masz typy węzłów
    mapping = {v['_nx_name']: i for i, v in zip(perm, g_ig.vs)}  # stare_id -> nowe_id
    return nx.relabel_nodes(G, mapping, copy=True)

def edge_list_to_graph6(edge_list: str, canonical: bool = False) -> str:
    """Convert edge-list to graph6."""
    G = edge_list_to_graph(edge_list)
    if canonical:
        G = canonize_graph(G)
    return graph_to_g6(G)

def graph6_to_edge_list(g6: str, canonical: bool = False) -> str:
    """Convert graph6 to edge-list."""
    G = g6_to_graph(g6)
    if canonical:
        G = canonize_graph(G)
    return graph_to_edge_list(G)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert between edge-list and graph6 representations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-g", "--to_graph6", action="store_true",
        help="Interpret input as edge list and output graph6")
    group.add_argument(
        "-e", "--to_edge_list", action="store_true",
        help="Interpret input as graph6 and output edge list")
    parser.add_argument(
        "-c", "--make_canonical", action="store_true",
        help="Canonicalize the graph before output")
    parser.add_argument(
        "graph", metavar="GRAPH",
        help="Input graph description")

    args = parser.parse_args()

    if args.to_graph6:
        output = edge_list_to_graph6(args.graph, canonical=args.make_canonical)
    else:  # to_edge_list
        output = graph6_to_edge_list(args.graph, canonical=args.make_canonical)

    print(output)
