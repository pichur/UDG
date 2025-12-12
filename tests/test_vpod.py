import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import unittest
import networkx as nx
from vpod import vertex_pair_orbits, process_graph

class TestVertexPairOrbits(unittest.TestCase):
    
    def test_p4_orbits(self):
        """Test vertex pair orbits for P4 (path graph with 4 vertices)"""
        # P4 in graph6 format: 1-3-0-2 (linear path)
        G = nx.from_graph6_bytes(b"CU")
        orbits = vertex_pair_orbits(G)
        
        self.assertEqual(len(orbits), 4)
        
        # Collect all pairs by type and distance
        self.assertEqual(orbits[0], ('n', 2, [(0,1), (2,3)]))
        self.assertEqual(orbits[1], ('E', 1, [(0,2), (1,3)]))
        self.assertEqual(orbits[2], ('E', 1, [(0,3)]))
        self.assertEqual(orbits[3], ('n', 3, [(1,2)]))
    
    def test_k4_orbits(self):
        """Test vertex pair orbits for K4 (complete graph with 4 vertices)"""
        # K4 in graph6 format: complete graph on 4 vertices
        G = nx.from_graph6_bytes(b"C~")
        orbits = vertex_pair_orbits(G)
        
        self.assertEqual(len(orbits), 1)
        
        # Collect all pairs by type and distance
        self.assertEqual(orbits[0], ('E', 1, [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]))
    
    def test_s4_orbits(self):
        """Test vertex pair orbits for S4 (star graph with 4 vertices)"""
        # S4 in graph6 format: star with center at vertex 0, leaves at 1,2,3,4
        G = nx.from_graph6_bytes(b"D?{")
        orbits = vertex_pair_orbits(G)
        
        self.assertEqual(len(orbits), 2)
        
        # Collect all pairs by type and distance
        self.assertEqual(orbits[0], ('n', 2, [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]))
        self.assertEqual(orbits[1], ('E', 1, [(0,4), (1,4), (2,4), (3,4)]))
    
    def test_disconnected_graph_orbits(self):
        """Test vertex pair orbits for disconnected graph"""
        # Create a disconnected graph: two separate edges
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        orbits = vertex_pair_orbits(G)
        
        # Should have orbits with infinite distance for disconnected pairs
        distances = [distance for _, distance, _ in orbits]
        self.assertIn(None, distances)  # None represents infinite distance
    
    def test_single_vertex_orbits(self):
        """Test vertex pair orbits for single vertex graph"""
        G = nx.Graph()
        G.add_node(0)
        orbits = vertex_pair_orbits(G)
        
        # Single vertex has no pairs
        self.assertEqual(len(orbits), 0)
    
    def test_two_vertex_connected_orbits(self):
        """Test vertex pair orbits for two connected vertices"""
        G = nx.Graph()
        G.add_edge(0, 1)
        orbits = vertex_pair_orbits(G)
        
        # Should have exactly one orbit with one edge
        self.assertEqual(len(orbits), 1)
        orbit_type, distance, pairs = orbits[0]
        self.assertEqual(orbit_type, "E")
        self.assertEqual(distance, 1)
        self.assertEqual(pairs, [(0, 1)])
    
    def test_two_vertex_disconnected_orbits(self):
        """Test vertex pair orbits for two disconnected vertices"""
        G = nx.Graph()
        G.add_nodes_from([0, 1])  # No edge between them
        orbits = vertex_pair_orbits(G)
        
        # Should have exactly one orbit with infinite distance
        self.assertEqual(len(orbits), 1)
        orbit_type, distance, pairs = orbits[0]
        self.assertEqual(orbit_type, "n")
        self.assertIsNone(distance)  # Infinite distance
        self.assertEqual(pairs, [(0, 1)])

    def test_process_graph_p4_u4(self):
        orbits = process_graph("4: 0,1 1,2 2,3", g6=False, unit=8, print_result=False, verbose=False)

        # Check the number of orbits
        self.assertEqual(len(orbits), 4)

        # Check the content of each orbit
        self.assertEqual(orbits[0], ('E', 1, [(0, 1), (2, 3)], (2,  7, True), (0,  8, True)))
        self.assertEqual(orbits[1], ('n', 2, [(0, 2), (1, 3)], (9, 13, True), (8, 15, True)))
        self.assertEqual(orbits[2], ('n', 3, [(0, 3)        ], (9, 20, True), (8, 23, True)))
        self.assertEqual(orbits[3], ('E', 1, [(1, 2)        ], (2,  7, True), (0,  8, True)))

    def test_process_graph_p4_u8(self):
        orbits = process_graph("4: 0,1 1,2 2,3", g6=False, unit=13, print_result=False, verbose=False)

        # Check the number of orbits
        self.assertEqual(len(orbits), 4)

        # Check the content of each orbit
        self.assertEqual(orbits[0], ('E', 1, [(0, 1), (2, 3)], ( 2, 12, True), ( 0, 13, True)))
        self.assertEqual(orbits[1], ('n', 2, [(0, 2), (1, 3)], (14, 23, True), (13, 25, True)))
        self.assertEqual(orbits[2], ('n', 3, [(0, 3)        ], (14, 35, True), (13, 38, True)))
        self.assertEqual(orbits[3], ('E', 1, [(1, 2)        ], ( 2, 12, True), ( 0, 13, True)))

if __name__ == "__main__":
    unittest.main()