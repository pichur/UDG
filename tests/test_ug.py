from asyncio import graph
import unittest
import networkx as nx
from ug import GraphUtil

class TestGraphUtil(unittest.TestCase):
        
    def test_reduce_single_node(self):
        """Test reducing a graph with a single node."""
        graph = nx.Graph()
        graph.add_node(0)
        
        result = GraphUtil.reduce(graph)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.output_canonical_g6, "@")
        self.assertEqual(result.reduced_nodes, 0)
        self.assertEqual(result.vertex_mapping, {})

    def test_reduce_two_disconnected_nodes(self):
        """Test reducing two disconnected nodes (no reduction should occur)."""
        graph = nx.Graph()
        graph.add_nodes_from([0, 1])
        
        result = GraphUtil.reduce(graph)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.output_canonical_g6, "A?")
        self.assertEqual(result.reduced_nodes, 0)
        self.assertEqual(result.vertex_mapping, {})

    def test_reduce_two_connected_nodes(self):
        """Test reducing two disconnected nodes (no reduction should occur)."""
        graph = nx.Graph()
        graph.add_nodes_from([0, 1])
        graph.add_edges_from([(0, 1)])

        result = GraphUtil.reduce(graph)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.output_canonical_g6, "@")
        self.assertEqual(result.reduced_nodes, 1)
        self.assertEqual(result.vertex_mapping, {0: [1]})

    def test_reduce_triangle(self):
        """Test reducing a triangle graph."""
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (0, 2), (1, 2)])  # Triangle
        
        result = GraphUtil.reduce(graph)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.output_canonical_g6, "@")
        self.assertEqual(result.reduced_nodes, 2)
        self.assertEqual(result.vertex_mapping, {0: [1, 2]})

    def test_reduce_path_graph(self):
        """Test reducing a path graph (no reduction should occur)."""
        graph = nx.path_graph(4)  # 0-1-2-3
        
        result = GraphUtil.reduce(graph)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.output_canonical_g6, "CR")
        self.assertEqual(result.reduced_nodes, 0)
        self.assertEqual(result.vertex_mapping, {})

    def test_reduce_star_graph(self):
        """Test reducing a star graph (leaves should be merged)."""
        graph = nx.star_graph(3)  # Center node 0 connected to 1, 2, 3
        
        result = GraphUtil.reduce(graph)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.output_canonical_g6, "CF")
        self.assertEqual(result.reduced_nodes, 0)
        self.assertEqual(result.vertex_mapping, {})

    def test_reduce_complex_graph(self):
        """Test reducing a complex graph."""
        graph = nx.Graph()
        # Create a graph where some nodes can be merged
        graph.add_edges_from([
            (0, 1), (0, 2),
            (1, 2),
            (2, 3), (2, 8),
            (3, 4), (3, 5), (3, 6),
            (4, 5), (4, 6), (4, 7),
            (5, 6), (5, 7),
            (6, 7),
            (7, 8), (7, 9)
        ])
        
        result = GraphUtil.reduce(graph)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.output_canonical_g6, "F?ErO")
        self.assertEqual(result.reduced_nodes, 3)
        self.assertEqual(result.vertex_mapping, {0: [1], 4: [5, 6]})

    def test_empty_graphs(self):
        """Test with empty graphs."""
        empty_graph = nx.Graph()
        single_node = nx.Graph()
        single_node.add_node(0)
        
        self.assertTrue (GraphUtil.contains_induced_subgraph(empty_graph, empty_graph))
        self.assertTrue (GraphUtil.contains_induced_subgraph(single_node, empty_graph))
        self.assertFalse(GraphUtil.contains_induced_subgraph(empty_graph, single_node))

    def test_single_node_subgraph(self):
        """Test finding single node as induced subgraph."""
        graph = nx.path_graph(3)  # 0-1-2
        single_node = nx.Graph()
        single_node.add_node(0)
        
        self.assertTrue(GraphUtil.contains_induced_subgraph(graph, single_node))

    def test_edge_as_subgraph(self):
        """Test finding an edge as induced subgraph."""
        graph = nx.path_graph(4)  # 0-1-2-3
        edge_graph = nx.Graph()
        edge_graph.add_edge(0, 1)
        
        self.assertTrue(GraphUtil.contains_induced_subgraph(graph, edge_graph))

    def test_triangle_in_complete_graph(self):
        """Test finding triangle in complete graph."""
        complete_4 = nx.complete_graph(4)
        triangle = nx.Graph()
        triangle.add_edges_from([(0, 1), (0, 2), (1, 2)])
        
        self.assertTrue(GraphUtil.contains_induced_subgraph(complete_4, triangle))

    def test_path_in_cycle(self):
        """Test finding path in cycle."""
        cycle_5 = nx.cycle_graph(5)  # 0-1-2-3-4-0
        path_3 = nx.path_graph(3)   # 0-1-2
        
        self.assertTrue(GraphUtil.contains_induced_subgraph(cycle_5, path_3))

    def test_star_not_in_path(self):
        """Test that star graph is not induced subgraph of path."""
        path_4 = nx.path_graph(4)  # 0-1-2-3
        star_3 = nx.star_graph(3)  # center connected to 3 leaves
        
        self.assertFalse(GraphUtil.contains_induced_subgraph(path_4, star_3))

    def test_triangle_not_in_path(self):
        """Test that triangle is not induced subgraph of path."""
        path_4 = nx.path_graph(4)  # 0-1-2-3 (no triangles)
        triangle = nx.Graph()
        triangle.add_edges_from([(0, 1), (0, 2), (1, 2)])
        
        self.assertFalse(GraphUtil.contains_induced_subgraph(path_4, triangle))

    def test_identical_graphs(self):
        """Test with identical graphs."""
        graph = nx.cycle_graph(5)
        identical_graph = nx.cycle_graph(5)
        
        self.assertTrue(GraphUtil.contains_induced_subgraph(graph, identical_graph))

    def test_larger_subgraph(self):
        """Test when subgraph is larger than main graph."""
        small_graph = nx.path_graph(3)
        large_graph = nx.path_graph(5)
        
        self.assertFalse(GraphUtil.contains_induced_subgraph(small_graph, large_graph))

    def test_disconnected_graphs(self):
        """Test with disconnected graphs."""
        # Main graph: two disconnected edges
        main_graph = nx.Graph()
        main_graph.add_edges_from([(0, 1), (2, 3)])
        
        # Subgraph: single edge
        sub_graph = nx.Graph()
        sub_graph.add_edge(0, 1)
        
        self.assertTrue(GraphUtil.contains_induced_subgraph(main_graph, sub_graph))
        
        # Subgraph: two disconnected nodes (should be found)
        two_nodes = nx.Graph()
        two_nodes.add_nodes_from([0, 1])
        
        self.assertTrue(GraphUtil.contains_induced_subgraph(main_graph, two_nodes))

    def test_complex_induced_subgraph(self):
        """Test with more complex graph structures."""
        # Create a graph that contains a specific pattern
        graph = nx.Graph()
        graph.add_edges_from([
            (0, 1), (1, 2), (2, 3), (3, 0),  # 4-cycle
            (1, 4), (2, 5)  # additional nodes
        ])
        
        # Look for the 4-cycle as induced subgraph
        cycle_4 = nx.cycle_graph(4)
        
        self.assertTrue(GraphUtil.contains_induced_subgraph(graph, cycle_4))

    def test_K23_in_K33(self):
        graphK23 = nx.Graph()
        graphK23.add_edges_from([
            (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4)  
        ])
        
        graphK33 = nx.Graph()
        graphK33.add_edges_from([
            (0, 1), (0, 3), (0, 5),
            (2, 1), (2, 3), (2, 5),
            (4, 1), (4, 3), (4, 5),
        ])
        
        self.assertTrue(GraphUtil.contains_induced_subgraph(graphK33, graphK23))

    def test_K23_not_in_K33_plus(self):
        graph_K23 = nx.Graph()
        graph_K23.add_edges_from([
            (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4)  
        ])
        
        graph_K33_plus = nx.Graph()
        graph_K33_plus.add_edges_from([
            (0, 1), (0, 3), (0, 5),
            (2, 1), (2, 3), (2, 5),
            (4, 1), (4, 3), (4, 5),
            (1, 3), (3, 5), (0, 4)
        ])
        
        self.assertFalse(GraphUtil.contains_induced_subgraph(graph_K33_plus, graph_K23))

    def test_star_5_6_7(self):
        graphS5 = nx.Graph()
        graphS5.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])

        graphS6 = nx.Graph()
        graphS6.add_edges_from([(0, 3), (1, 3), (2, 3), (3, 4), (3, 5), (3, 6)])
        
        graphS7 = nx.Graph()
        graphS7.add_edges_from([(0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)])
        
        
        self.assertTrue (GraphUtil.contains_induced_subgraph(graphS5, graphS5))
        self.assertFalse(GraphUtil.contains_induced_subgraph(graphS5, graphS6))
        self.assertFalse(GraphUtil.contains_induced_subgraph(graphS5, graphS7))

        self.assertTrue (GraphUtil.contains_induced_subgraph(graphS6, graphS5))
        self.assertTrue (GraphUtil.contains_induced_subgraph(graphS6, graphS6))
        self.assertFalse(GraphUtil.contains_induced_subgraph(graphS6, graphS7))

        self.assertTrue (GraphUtil.contains_induced_subgraph(graphS7, graphS5))
        self.assertTrue (GraphUtil.contains_induced_subgraph(graphS7, graphS6))
        self.assertTrue (GraphUtil.contains_induced_subgraph(graphS7, graphS7))



if __name__ == '__main__':
    unittest.main()