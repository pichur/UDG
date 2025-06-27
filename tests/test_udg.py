import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import unittest
from udg import Graph, udg_recognition
from Graph6Converter import Graph6Converter

class TestUDG(unittest.TestCase):
    def test_triangle(self):
        g = Graph(3)
        g.add_edge(0,1)
        g.add_edge(1,2)
        g.add_edge(0,2)
        self.assertTrue(udg_recognition(g))

    def test_path_four(self):
        g = Graph(4)
        g.add_edge(0,1)
        g.add_edge(1,2)
        g.add_edge(2,3)
        self.assertTrue(udg_recognition(g))

    def test_non_udg(self):
        non_udf = [
            '5:1,2;1,3;1,4;2,5;3,5;4,5',
            '6:1,2;1,3;2,4;3,4;2,5;3,6;5,6',
            '7:1,2;1,3;1,4;1,5;1,6;1,7',
            '7:1,2;1,3;2,4;3,4;2,5;3,7;5,6;6,7',
            '7:1,2;1,3;1,4;2,4;2,5;3,6;4,7;5,6;6,7',
            '7:1,2;1,3;1,4;2,5;2,7;3,6;4,7;5,6;6,7',
            '7:1,2;1,4;2,3;3,4;2,5;3,6;4,7;5,6;6,7'
        ]

        for graph in non_udf:
            nxg = Graph6Converter._parse_edge_list(graph)
            g = Graph(nxg)
            self.assertFalse(udg_recognition(g), graph)

if __name__ == '__main__':
    unittest.main()
