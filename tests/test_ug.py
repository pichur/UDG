import unittest
import networkx as nx
from ug import GraphUtil

class TestGraphUtilReduce(unittest.TestCase):
        
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

if __name__ == '__main__':
    unittest.main()