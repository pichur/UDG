import argparse
import time
import Graph6Converter
from discrete_disk import MODE_O, MODE_B, MODE_I
from udg import Graph

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

def vertex_pair_orbits(G):
    """
    Returns a list of vertex pair orbits.
    Each orbit is a tuple (orbit_type, distance, orbit_pairs) where:
    - orbit_type is 'E' for edges or 'n' for non-edges
    - distance is the shortest path distance between vertices (or None if disconnected)
    - orbit_pairs is a sorted list of pairs (u,v) with u<v
    """
    # All automorphisms of G (as dictionaries "old_node -> new_node")
    autos = list(GraphMatcher(G, G).isomorphisms_iter())

    # All vertex pairs in normalized form (u<v)
    vertices = list(G.nodes())
    all_pairs = [(u, v) for i, u in enumerate(vertices) for v in vertices[i+1:]]
    
    seen = set()
    orbits = []

    for pair in all_pairs:
        if pair in seen:
            continue
        # Collect all images of the pair under automorphisms
        orb = set()
        for f in autos:
            u, v = pair
            mu, mv = f[u], f[v]
            orb.add(tuple(sorted((mu, mv))))
        # Save normalized orbit with type classification and distance
        orb_list = sorted(orb)
        # Classify orbit type based on first pair (all pairs in orbit have same type)
        u, v = pair
        orbit_type = "E" if G.has_edge(u, v) else "n"
        
        # Calculate shortest path distance
        try:
            distance = nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            distance = None  # Vertices are not connected
        
        orbits.append((orbit_type, distance, orb_list))
        seen.update(orb)
    return orbits

def print_vertex_pair_orbits(orbits):
    """
    Prints all vertex pair orbits for a given graph in a formatted way.
    Each orbit is expected to be a tuple (orbit_type, distance, orbit_pairs).
    """
    for i, (orbit_type, distance, orbit_pairs) in enumerate(orbits):
        orbit_letter = chr(ord('a') + i)
        distance_str = str(distance) if distance is not None else "∞"
        print(f"  Orbit {orbit_letter} ({orbit_type}, d={distance_str}): {orbit_pairs}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate distances of vertex pair orbits in udg graph.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-g", "--graph6", action="store_true",
        help="Check graph given as graph6")
    group.add_argument(
        "-e", "--edge_list", action="store_true",
        help="Check graph given as edge list")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output for debugging")
    parser.add_argument(
        "-u", "--unit", type=int,
        help="Start unit")
    parser.add_argument(
        "-a", "--order", type=str, default="DD",
        help="Order mode")
    parser.add_argument(
        "graph", metavar="GRAPH", nargs="?", default="",
        help="Input graph description (multiple graphs can be separated by ;)")

    args = parser.parse_args()

    # Split input by semicolons for multiple graphs
    if ';' in args.graph:
        graphs_input = [g.strip() for g in args.graph.split(';') if g.strip()]
    else:
        graphs_input = [args.graph] if args.graph else []

    if not graphs_input:
        print("No graph provided")
        return

    for i, graph_input in enumerate(graphs_input):
        if len(graphs_input) > 1:
            print(f"\n=== Processing Graph {i+1}/{len(graphs_input)} ===")
            print(f"Input: {graph_input}")

        if args.graph6:
            g = Graph(Graph6Converter.g6_to_graph(graph_input))
        elif args.edge_list:
            g = Graph(Graph6Converter.edge_list_to_graph(graph_input))

        g.set_verbose(args.verbose)

        if args.unit:
            g.set_unit(args.unit)

        if args.order:
            g.order_mode = args.order

        g.last_verbose_time = time.time()
        g.calculate_vertex_distances()

        g.print_vertex_distances()

if __name__ == "__main__":
    main()
