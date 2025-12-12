import argparse
import time
import Graph6Converter
from ug import GraphUtil
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

def print_vertex_pair_orbit(print_result:bool, output_file:str, orbit, i=None):
    """
    Prints a single vertex pair orbit for a given graph in a formatted way.
    Each orbit is expected to be a tuple (orbit_type, distance, orbit_pairs).
    """
    orbit_type, distance, orbit_pairs = orbit
    orbit_letter = (' ' + chr(ord('a') + i)) if i is not None else ''
    distance_str = str(distance) if distance is not None else "∞"
    out(print_result, output_file, f"  Orbit{orbit_letter} ({orbit_type}, d={distance_str}): {orbit_pairs}")

def print_vertex_pair_orbits(print_result:bool, output_file:str, orbits):
    """
    Prints all vertex pair orbits for a given graph in a formatted way.
    """
    for i, orbit in enumerate(orbits):
        print_vertex_pair_orbit(print_result, output_file, orbit, i)

def get_ranges(values:list[int], border:bool, ignore_range_check:bool=False) -> tuple[int,int,bool]|None:
    """Get min and max indexes of a specific value in a list, ensuring all intermediate values are the same."""
    search_values = [MODE_I, MODE_B] if border else [MODE_I]
    min_index = max_index = None
    for i, val in enumerate(values):
        if val in search_values:
            if min_index is None:
                min_index = i
            max_index = i
    
    if min_index is None:
        return None
    
    continous = True
    # Check if there are any different values between min and max indices
    if min_index is not None and max_index is not None and min_index != max_index:
        for i in range(min_index + 1, max_index):
            if values[i] not in search_values:
                if ignore_range_check:
                    continous = False
                else:
                    raise ValueError(f"Found different value {values[i]} at index {i} between min index {min_index} and max index {max_index}")
    
    return (min_index, max_index, continous)

def process_graph(graph_input:str, g6:bool=True, unit:int=4, order:str="DD", ignore_range_check:bool=False, print_result:bool=False, output_file:str="", chars:str="█▒ ", verbose:bool=False, limit_negative_distances:bool=False):
    start_time = time.time()
    
    if g6:
        g:nx.Graph = Graph6Converter.g6_to_graph(graph_input)
    else:
        g:nx.Graph = Graph6Converter.edge_list_to_graph(graph_input)

    udg = Graph(g)

    reduction_info = GraphUtil.reduce(g)
    if reduction_info.reduced_nodes > 0:
        out(print_result, output_file, f"Graph reduced to : {reduction_info.output_canonical_g6}")
        return 

    udg.last_verbose_time = time.time()
    udg.set_verbose(verbose)

    udg.limit_negative_distances = limit_negative_distances

    if unit:
        udg.set_unit(unit)

    orbits = vertex_pair_orbits(g)

    print_vertex_pair_orbits(print_result, output_file, orbits)

    header = udg.get_node_distances_info(True)
    out(print_result, output_file, f"\n     :       : {header}       Time")
    
    result = []
    for i, orbit in enumerate(orbits):
        iteration_start_time = time.time()
        orbit_letter = chr(ord('a') + i)
        orbit_type, distance, orbit_pairs = orbit
        nodes = orbit_pairs[0]
        node_distances = udg.calculate_node_distances(nodes, order)

        range_i = get_ranges(node_distances, False, ignore_range_check)
        range_b = get_ranges(node_distances, True , ignore_range_check)
        result.append((orbit_type, distance, orbit_pairs, range_i, range_b))
        iteration_time = time.time() - iteration_start_time
        if print_result or output_file:
            msg = udg.get_node_distances_info(False, chars)
            u, v = nodes
            edge_indicator = '-' if orbit_type == 'E' else ' '
            default_range_b = (0 if distance == 1 else unit, distance * unit)
            short_range_b = None if range_b is None else (range_b[0], range_b[1])
            not_default_range_b = short_range_b != default_range_b
            not_continueous_range_i = range_i is not None and not range_i[2]
            not_continueous_range_b = range_b is not None and not range_b[2]
            attention_mark = '  !!!' if not_default_range_b or not_continueous_range_i or not_continueous_range_b else ''
            range_mark = (' ' + ('R' if distance == 1 else 'r')) if not_default_range_b else ''
            continueous_i_mark = ' I' if not_continueous_range_i else ''
            continueous_b_mark = ' B' if not_continueous_range_b else ''
            out(print_result, output_file, f"{orbit_letter} [{distance}]: {u} {edge_indicator} {v} : {msg}    {iteration_time:7.2f} s{attention_mark}{range_mark}{continueous_i_mark}{continueous_b_mark}")

    stop_time = time.time()
    out(print_result, output_file, f"\nTime taken: {stop_time - start_time:.4f} s")

    return result

def out(print_to_console:bool, output_file: str, msg: str) -> None:
    if print_to_console:
        print(msg)
    if output_file:
        with open(output_file, 'a') as f:
            f.write(msg + '\n')

def main() -> None:
    # abcdefghijklmnopqrstuvwxyz
    # -b-d---h-jkl-n--qrst--wxyz
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
        "-f", "--file", action="store_true",
        help="Read graphs from file (one graph6 per line)")
    parser.add_argument(
        "-o", "--output", type=str,
        help="Write output to file")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output for debugging")
    parser.add_argument(
        "-p", "--print", action="store_true",
        help="Print output")
    parser.add_argument(
        "-c", "--chars", default="█▒ ",
        help="Default characters for print ranges")
    parser.add_argument(
        "-i", "--ignore", action="store_true",
        help="Ignore ranges check")
    parser.add_argument(
        "-u", "--unit", type=int,
        help="Working unit")
    parser.add_argument(
        "-a", "--order", type=str, default="DD",
        help="Order mode")
    parser.add_argument(
        "-m", "--limit_negative_distances", action="store_true",
        help="Turn on limiting negative distances")
    parser.add_argument(
        "graph", metavar="GRAPH", nargs="?", default="",
        help="Input graph description (multiple graphs can be separated by ;) or file if -f is used)")

    args = parser.parse_args()

    if args.file:
        with open(args.graph, 'r') as f:
            graphs_input = [line.strip() for line in f if line.strip()]
    else:
        # Split input by semicolons for multiple graphs
        if ';' in args.graph:
            graphs_input = [g.strip() for g in args.graph.split(';') if g.strip()]
        else:
            graphs_input = [args.graph] if args.graph else []

    if not graphs_input:
        print("No graph provided")
        return

    for i, graph_input in enumerate(graphs_input):
        out(args.print, args.output, f"\n=== Processing Graph {i+1}/{len(graphs_input)} ===")
        out(args.print, args.output, f"Input: {graph_input}")
        process_graph(graph_input, args.graph6, args.unit, args.order, args.ignore, args.print, args.output, args.chars, args.verbose, args.limit_negative_distances)

if __name__ == "__main__":
    main()
