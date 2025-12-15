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
    
    class ReduceInfo:
        def __init__(self, input_g6:str, input_canonical_g6:str, reduced_nodes:int, output_canonical:nx.Graph, output_canonical_g6:str, vertex_mapping:dict):
            self.input_g6 = input_g6
            self.input_canonical_g6 = input_canonical_g6
            self.reduced_nodes = reduced_nodes
            self.output_canonical = output_canonical
            self.output_canonical_g6 = output_canonical_g6
            self.vertex_mapping = vertex_mapping

    @classmethod
    def contains_induced_subgraph(cls, graph:nx.Graph, subgraph:nx.Graph) -> bool:
        """
        Check if the graph contains the given subgraph as an induced subgraph.
        
        Args:
            graph    (nx.Graph): The main graph
            subgraph (nx.Graph): The subgraph to check for
        Returns:
            bool: True if subgraph is an induced subgraph of graph, False otherwise
        """
        gm = nx.algorithms.isomorphism.GraphMatcher(graph, subgraph)
        return gm.subgraph_is_isomorphic()

    @classmethod
    def reduce(cls, graph: nx.Graph) -> ReduceInfo:
        """
        Reduce graph by merging nodes with identical neighborhoods.
        Keeps the node with smaller index and returns canonical form.

        Args:
            graph (nx.Graph): The input graph to reduce.

        Returns:
            dict|None: A dictionary with 'canonical_g6' and 'vertex_mapping' or None if reduction fails.
        """
        # Create a copy to avoid modifying original
        reduced_graph = graph.copy()
        vertex_mapping = {}  # Maps remaining vertices to lists of merged vertices

        changed = True
        while changed:
            changed = False
            nodes = list(reduced_graph.nodes())
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node1, node2 = nodes[i], nodes[j]
                    
                    # Skip if either node was already removed
                    if node1 not in reduced_graph or node2 not in reduced_graph:
                        continue
                        
                    # Check if nodes are connected and have identical neighborhoods
                    if reduced_graph.has_edge(node1, node2):
                        neighbors1 = set(reduced_graph.neighbors(node1)) - {node2}
                        neighbors2 = set(reduced_graph.neighbors(node2)) - {node1}
                        
                        if neighbors1 == neighbors2:
                            # Keep node with smaller index, remove the other
                            keep_node = min(node1, node2)
                            remove_node = max(node1, node2)
                            
                            # Update vertex mapping - create entries only when reduction happens
                            if keep_node not in vertex_mapping:
                                vertex_mapping[keep_node] = []
                            vertex_mapping[keep_node].append(remove_node)
                            
                            # Remove the node with larger index
                            reduced_graph.remove_node(remove_node)
                            changed = True
                            break
                if changed:
                    break

        output_canonical_g6 = Graph6Converter.graph_to_g6(reduced_graph, canonical=True)

        return GraphUtil.ReduceInfo(
            input_g6=Graph6Converter.graph_to_g6(graph),
            input_canonical_g6=Graph6Converter.graph_to_g6(graph, canonical=True),
            reduced_nodes=graph.number_of_nodes() - reduced_graph.number_of_nodes(),
            output_canonical=Graph6Converter.g6_to_graph(output_canonical_g6),
            output_canonical_g6=output_canonical_g6,
            vertex_mapping=vertex_mapping
        )

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