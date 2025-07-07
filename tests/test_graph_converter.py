import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import glob
import unittest
import networkx as nx
import Graph6Converter
import subprocess


class TestGraph6Converter(unittest.TestCase):
    def test_roundtrip_example(self):
        text = "5: 1,2 ; 1,3 ; 1,4 ; 2,5 ; 3,5 ; 4,5"
        g6 = Graph6Converter.edge_list_to_graph6(text)
        back = Graph6Converter.graph6_to_edge_list(g6)
        self.assertEqual(g6, Graph6Converter.edge_list_to_graph6(back))

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
                    edges = Graph6Converter.graph6_to_edge_list(g6)
                    g6_again = Graph6Converter.edge_list_to_graph6(edges)
                    G1 = nx.from_graph6_bytes(g6.encode())
                    G2 = nx.from_graph6_bytes(g6_again.encode())
                    self.assertTrue(nx.is_isomorphic(G1, G2))

    def test_cli_to_graph6(self):
        text = "3: 1,2 ; 2,3"
        expected = Graph6Converter.edge_list_to_graph6(text, canonical=False)
        script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Graph6Converter.py')
        res = subprocess.run([sys.executable, script, '-g', text], capture_output=True, text=True)
        self.assertEqual(res.stdout.strip(), expected)

    def test_cli_to_edge_list_canonical(self):
        g6 = Graph6Converter.edge_list_to_graph6("4: 1,2 ; 2,3 ; 3,4")
        script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Graph6Converter.py')
        res = subprocess.run([sys.executable, script, '-e', '-c', g6], capture_output=True, text=True)
        expected = Graph6Converter.graph6_to_edge_list(g6, canonical=True)
        self.assertEqual(res.stdout.strip(), expected)

if __name__ == '__main__':
    unittest.main()
