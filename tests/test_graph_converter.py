import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import glob
import unittest
import networkx as nx
from graph_converter import Graph6Converter


class TestGraph6Converter(unittest.TestCase):
    def setUp(self):
        self.conv = Graph6Converter()

    def test_roundtrip_example(self):
        text = "5: 1,2 ; 1,3 ; 1,4 ; 2,5 ; 3,5 ; 4,5"
        g6 = self.conv.edge_list_to_graph6(text)
        back = self.conv.graph6_to_edge_list(g6)
        self.assertEqual(g6, self.conv.edge_list_to_graph6(back))

    def test_data_graphs_roundtrip(self):
        data_files = glob.glob(os.path.join('data', '*.g6.txt'))
        for fname in data_files:
            with open(fname) as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    g6 = line.strip()
                    if not g6:
                        continue
                    edges = self.conv.graph6_to_edge_list(g6)
                    g6_again = self.conv.edge_list_to_graph6(edges)
                    G1 = nx.from_graph6_bytes(g6.encode())
                    G2 = nx.from_graph6_bytes(g6_again.encode())
                    self.assertTrue(nx.is_isomorphic(G1, G2))

if __name__ == '__main__':
    unittest.main()
