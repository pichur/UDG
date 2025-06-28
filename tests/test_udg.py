import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import unittest
from udg import Graph, udg_recognition
from Graph6Converter import Graph6Converter

class TestUDG(unittest.TestCase):
    def test_udg_3(self):
        self._check_udg(['3:1,2;2,3;3,1'], True)

    def test_udg_4(self):
        self._check_udg([
            '4:1,2;2,3;3,4',
            '4:1,2;2,3;3,4;4,1',
            '4:1,2;2,3;3,4;4,1;1,3',
            '4:1,2;2,3;3,4;4,1;1,3;2,4',
            ], True)

    def test_non_udg_5(self):
        self._check_udg(['5:1,2;1,3;1,4;2,5;3,5;4,5'], False)

    def test_non_udg_6(self):
        self._check_udg(['6:1,2;1,3;2,4;3,4;2,5;3,6;5,6'], False)
        
    def test_non_udg_7(self):
        self._check_udg([
            '7:1,2;1,3;1,4;1,5;1,6;1,7',
            '7:1,2;1,3;2,4;3,4;2,5;3,7;5,6;6,7',
            '7:1,2;1,3;1,4;2,4;2,5;3,6;4,7;5,6;6,7',
            '7:1,2;1,3;1,4;2,5;2,7;3,6;4,7;5,6;6,7',
            '7:1,2;1,4;2,3;3,4;2,5;3,6;4,7;5,6;6,7'
            ], False)
        
    def _check_udg(self, graphs: list[str], expected: bool):
        for graph in graphs:
            print(f"Testing non-UDG graph: {graph}")
            nxg = Graph6Converter._parse_edge_list(graph)
            g = Graph(nxg)
            self.assertEqual(expected, udg_recognition(g), graph)

if __name__ == '__main__':
    unittest.main()
