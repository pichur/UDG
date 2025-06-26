import itertools
import networkx as nx
from networkx.readwrite.graph6 import _generate_graph6_bytes


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

    def edge_list_to_graph6(self, text: str) -> str:
        G = self._parse_edge_list(text)
        return self._canonical_graph6(G)

    def graph6_to_edge_list(self, g6: str) -> str:
        G = nx.from_graph6_bytes(g6.strip().encode())
        return self._graph_to_edge_list(G)
