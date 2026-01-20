import argparse
import Graph6Converter
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from ug import GraphUtil


def get_vertex_orbits(g):
    """
    Get vertex orbits using NetworkX automorphisms.
    Returns a list of sets, where each set contains vertices in the same orbit.
    """
    # Get all automorphisms
    autos = list(GraphMatcher(g, g).isomorphisms_iter())
    
    vertices = list(g.nodes())
    seen = set()
    orbits = []
    
    for vertex in vertices:
        if vertex in seen:
            continue
            
        # Find orbit of this vertex
        orbit = set()
        for auto in autos:
            orbit.add(auto[vertex])
        
        orbits.append(orbit)
        seen.update(orbit)
    
    return orbits


def process_graph_file(input_file: str, output_file: str) -> None:
    """
    Process graphs from input file and write results to output file.
    
    Args:
        input_file: Path to input file with one g6 graph per line
        output_file: Path to output file with formatted results
    """

    graph_K23 = nx.Graph()
    graph_K23.add_edges_from([
        (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4)  
    ])

    graph_S6 = nx.Graph()
    graph_S6.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)])

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        # Write header
        f_out.write(f"{'Input':>10} {'Canonical':>10} {'Reduced':>10}\n")
        f_out.write("-" * 32 + "\n")
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            
            g6 = line
            try:
                # Parse input graph
                g = Graph6Converter.g6_to_graph(g6)
                
                # Get canonical form
                canonical_g6 = Graph6Converter.graph_to_g6(g, canonical=True)
                
                # Get reduced form
                reduction_info = GraphUtil.reduce(g)
                reduced_g6 = reduction_info.output_canonical_g6

                marker = '-'

                if g6 != canonical_g6:
                    marker += 'C'
                else:
                    marker += '_'

                marker += '-'

                if reduction_info.reduced_nodes == 0:
                    marker += 'R'
                else:
                    marker += '_'
                
                marker += '-'

                # Get vertex orbits
                orbits = get_vertex_orbits(g)
                orbit_str = ",".join(f"({','.join(map(str, sorted(orbit)))})" for orbit in sorted(orbits, key=lambda x: min(x)))

                # Get edges from the graph
                edges = list(Graph6Converter.g6_to_graph(canonical_g6).edges())
                
                f_out.write(f"{line:>10} {canonical_g6:>10} {reduced_g6:>10} {marker} {len(orbits)} {orbit_str:<20} {edges}\n")
                
            except Exception as e:
                print(f"Error processing line {line_num}: '{line}' - {e}")
                # Write error indicator
                f_out.write(f"{line:>10} {'ERROR':>10} {'ERROR':>10}\n")

def main() -> None:
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Pre-process graphs: read g6 from file, output canonical and reduced forms.")
    
    parser.add_argument(
        "input_file", metavar="INPUT_FILE",
        help="Input file with one g6 graph per line")
    
    parser.add_argument(
        "output_file", metavar="OUTPUT_FILE",
        help="Output file with formatted results")
    
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        if args.verbose:
            print(f"Processing graphs from: {args.input_file}")
            print(f"Writing results to: {args.output_file}")
        
        process_graph_file(args.input_file, args.output_file)
        
        if args.verbose:
            print("Processing completed successfully")
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()