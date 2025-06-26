import itertools
import argparse
import networkx as nx
from networkx.readwrite.graph6 import _generate_graph6_bytes, to_graph6_bytes


class Graph6Converter:
    """Convert between custom edge-list format and canonical graph6."""

    @staticmethod
    def _parse_edge_list(text: str) -> nx.Graph:
        """Parse edge-list formatted as 'n: u,v ; x,y ...'."""
        text = text.strip()
        if not text:
            raise ValueError("Empty graph description")
        parts = text.split(":", 1)
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

    @staticmethod
    def _graph_to_edge_list(G: nx.Graph) -> str:
        n = len(G)
        edges = sorted({tuple(sorted((u + 1, v + 1))) for u, v in G.edges()})
        edges_str = " ; ".join(f"{u},{v}" for u, v in edges)
        return f"{n}: {edges_str}" if edges_str else f"{n}:"

    @staticmethod
    def _canonical_graph6(G: nx.Graph) -> str:
        nodes = list(G.nodes())
        best = None
        for perm in itertools.permutations(nodes):
            mapping = {perm[i]: i for i in range(len(perm))}
            H = nx.relabel_nodes(G, mapping)
            g6 = b"".join(_generate_graph6_bytes(H, range(len(perm)), False)).decode().strip()
            if best is None or g6 < best:
                best = g6
        return best

    def edge_list_to_graph6(self, text: str, canonical: bool = True) -> str:
        """Convert edge-list text to graph6 string."""
        G = self._parse_edge_list(text)
        if canonical:
            return self._canonical_graph6(G)
        return to_graph6_bytes(G, nodes=sorted(G.nodes()), header=False).decode().strip()

    def graph6_to_edge_list(self, g6: str) -> str:
        G = nx.from_graph6_bytes(g6.strip().encode())
        return self._graph_to_edge_list(G)

    def canonicalize_graph6(self, g6: str) -> str:
        """Return canonical graph6 representation for given graph6 string."""
        G = nx.from_graph6_bytes(g6.strip().encode())
        return self._canonical_graph6(G)

    def canonicalize_edge_list(self, text: str) -> str:
        """Return canonical edge-list representation for given edge-list text."""
        g6 = self.edge_list_to_graph6(text, canonical=True)
        return self.graph6_to_edge_list(g6)


def main() -> None:
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

    conv = Graph6Converter()

    if args.to_graph6:
        if args.make_canonical:
            output = conv.edge_list_to_graph6(args.graph, canonical=True)
        else:
            output = conv.edge_list_to_graph6(args.graph, canonical=False)
    else:  # to_edge_list
        if args.make_canonical:
            canon = conv.canonicalize_graph6(args.graph)
            output = conv.graph6_to_edge_list(canon)
        else:
            output = conv.graph6_to_edge_list(args.graph)

    print(output)


if __name__ == "__main__":
    main()

