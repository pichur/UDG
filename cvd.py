import argparse
import time
import Graph6Converter
from discrete_disk import MODE_O, MODE_B, MODE_I
from udg import Graph

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate distances of vertex in udg graph.")
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
        "-n", "--not_limit_points", action="store_true",
        help="Turn off limiting points")
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

        g.set_log_level(args.verbose)

        if args.unit:
            g.set_unit(args.unit)

        if args.order:
            g.order_mode = args.order

        if args.not_limit_points:
            g.limit_points = False

        g.last_verbose_time = time.time()
        g.calculate_vertex_distances()

        g.print_vertex_distances()

if __name__ == "__main__":
    main()
