import argparse
import networkx as nx
from networkx import to_graph6_bytes
import Graph6Converter
import numpy as np
import igraph as ig

class GraphUtil:
    """
    Class for working with graphs with multiple representations and canonization.
    """
    
    @classmethod
    def to_adjacency_matrix(cls, graph):
        return nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    
    @classmethod
    def to_incidence_matrix(cls, graph):
        return nx.incidence_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    
    @classmethod
    def display_all_representations(cls, graph):
        """
        Display all representations of the graph.
        
        Args:
            show_canonical: If True, also show canonical representations
        """
        print(f"Graph with {graph.number_of_nodes()} vertices and {graph.number_of_edges()} edges")
        print()
        
        # Original representations
        print("=== Original Graph ===")
        print(f"Graph6: {Graph6Converter.graph_to_g6(graph)}")
        print(f"Edge list: {Graph6Converter.graph_to_edge_list(graph)}")

        print("\nAdjacency Matrix:")
        adj_matrix = cls.to_adjacency_matrix(graph)
        cls._print_matrix(adj_matrix, "Adjacency")

    @classmethod
    def _print_matrix(cls, matrix, matrix_type):
        """Helper method to print matrices in a formatted way."""
        if matrix.size == 0:
            print(f"{matrix_type} matrix is empty")
            return
        
        rows, cols = matrix.shape
        if rows <= 10 and cols <= 10:  # Print full matrix for small graphs
            for row in matrix:
                print("  " + " ".join(f"{int(val):2d}" for val in row))
        else:  # Print summary for large graphs
            print(f"  {matrix_type} matrix: {rows}x{cols} (too large to display)")
    
    @classmethod
    def get_graph_properties(cls, graph):
        """
        Get basic properties of the graph.
        
        Returns:
            dict: Dictionary with graph properties
        """
        return {
            'vertices': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'is_connected': nx.is_connected(graph),
            'density': nx.density(graph),
            'diameter': nx.diameter(graph) if nx.is_connected(graph) else None,
            'clustering': nx.average_clustering(graph),
            'degree_sequence': sorted([d for n, d in graph.degree()], reverse=True)
        }


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Work with undirected graphs in various representations.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-g", "--graph6", action="store_true",
        help="Input graph as graph6 format")
    group.add_argument(
        "-e", "--edge_list", action="store_true",
        help="Input graph as edge list format")
    
    parser.add_argument(
        "-c", "--canonical", action="store_true",
        help="Show canonical representations")
    parser.add_argument(
        "-p", "--properties", action="store_true",
        help="Show graph properties")
    parser.add_argument(
        "graph", metavar="GRAPH",
        help="Input graph description")
    
    args = parser.parse_args()
    
    if not args.graph:
        print("No graph provided")
        return
    
    print(f"Input: {args.graph}")

    try:
        # Create graph based on input type
        if args.graph6:
            g = Graph6Converter.g6_to_graph(args.graph)
        elif args.edge_list:
            g = Graph6Converter.edge_list_to_graph(args.graph)
        
        # Display all representations
        GraphUtil.display_all_representations(g)
        
        if (args.canonical):
            print("\n=== Canonical Representations ===")
            GraphUtil.display_all_representations(Graph6Converter.canonize_graph(g))

        # Show properties if requested
        if args.properties:
            print("\n=== Graph Properties ===")
            props = g.get_graph_properties()
            for key, value in props.items():
                if key == 'degree_sequence':
                    print(f"{key}: {value}")
                else:
                    print(f"{key}: {value}")
    
    except Exception as e:
        print(f"Error processing graph '{args.graph}': {e}")


if __name__ == "__main__":
    main()