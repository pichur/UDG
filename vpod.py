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

def print_vertex_pair_orbit(orbit, i=None):
    """
    Prints a single vertex pair orbit for a given graph in a formatted way.
    Each orbit is expected to be a tuple (orbit_type, distance, orbit_pairs).
    """
    orbit_type, distance, orbit_pairs = orbit
    orbit_letter = (' ' + chr(ord('a') + i)) if i is not None else ''
    distance_str = str(distance) if distance is not None else "∞"
    print(f"  Orbit{orbit_letter} ({orbit_type}, d={distance_str}): {orbit_pairs}")

def print_vertex_pair_orbits(orbits):
    """
    Prints all vertex pair orbits for a given graph in a formatted way.
    """
    for i, orbit in enumerate(orbits):
        print_vertex_pair_orbit(orbit, i)

def get_min_max_indexes(values:list[int], value:int):
    """Get min and max indexes of a specific value in a list, ensuring all intermediate values are the same."""
    min_index = max_index = None
    for i, val in enumerate(values):
        if val == value:
            if min_index is None:
                min_index = i
            max_index = i
    
    if min_index is None:
        return None
    
    # Check if there are any different values between min and max indices
    if min_index is not None and max_index is not None and min_index != max_index:
        for i in range(min_index + 1, max_index):
            if values[i] != value:
                if value == MODE_B and values[i] == MODE_I:
                    # Allow MODE_I between MODE_B indices
                    continue
                else:
                    raise ValueError(f"Found different value {values[i]} at index {i} between min index {min_index} and max index {max_index}")
    
    return (min_index, max_index)
        
def process_graph(graph_input:str, g6:bool=True, unit:int=4, print_result:bool=False, verbose:bool=False):
    start_time = time.time()
    
    if g6:
        g:nx.Graph = Graph6Converter.g6_to_graph(graph_input)
    else:
        g:nx.Graph = Graph6Converter.edge_list_to_graph(graph_input)

    udg = Graph(g)
    udg.last_verbose_time = time.time()
    udg.set_verbose(verbose)

    if unit:
        udg.set_unit(unit)

    orbits = vertex_pair_orbits(g)

    if print_result:
        print_vertex_pair_orbits(orbits)

    if print_result:
        header = udg.get_node_distances_info(True)
        print(f"\n     :       : {header}    Time")
    
    result = []
    for i, orbit in enumerate(orbits):
        iteration_start_time = time.time()
        orbit_letter = chr(ord('a') + i)
        orbit_type, distance, orbit_pairs = orbit
        nodes = orbit_pairs[0]
        node_distances = udg.calculate_node_distances(nodes)

        result.append((orbit_type, distance, orbit_pairs, get_min_max_indexes(node_distances, MODE_I), get_min_max_indexes(node_distances, MODE_B)))
        iteration_time = time.time() - iteration_start_time
        if print_result:
            msg = udg.get_node_distances_info()
            u, v = nodes
            edge_indicator = '-' if orbit_type == 'E' else ' '
            print(f"{orbit_letter} [{distance}]: {u} {edge_indicator} {v} : {msg}    {iteration_time:.4f} s")

    stop_time = time.time()
    if print_result:
        print(f"\nTime taken: {stop_time - start_time:.4f} s")

    return result

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
        "-p", "--print", action="store_true",
        help="Print output")
    parser.add_argument(
        "-u", "--unit", type=int,
        help="Start unit")
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
        if print:
            print(f"\n=== Processing Graph {i+1}/{len(graphs_input)} ===")
            print(f"Input: {graph_input}")
        process_graph(graph_input, args.graph6, args.unit, args.print, args.verbose)

if __name__ == "__main__":
    main()
