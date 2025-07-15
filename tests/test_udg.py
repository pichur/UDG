import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import unittest
from udg import Graph
import Graph6Converter

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

    @unittest.skip("Test disabled")
    def test_non_udg_5(self):
        self._check_udg(['5:1,2;1,3;1,4;2,5;3,5;4,5'], False)

    @unittest.skip("Test disabled")
    def test_non_udg_6(self):
        self._check_udg(['6:1,2;1,3;2,4;3,4;2,5;3,6;5,6'], False)
        
    @unittest.skip("Test disabled")
    def test_non_udg_7(self):
        self._check_udg([
            '7:1,2;1,3;1,4;1,5;1,6;1,7',
            '7:1,2;1,3;2,4;3,4;2,5;3,7;5,6;6,7',
            '7:1,2;1,3;1,4;2,4;2,5;3,6;4,7;5,6;6,7',
            '7:1,2;1,3;1,4;2,5;2,7;3,6;4,7;5,6;6,7',
            '7:1,2;1,4;2,3;3,4;2,5;3,6;4,7;5,6;6,7'
            ], False)
        
    def _check_udg(self, graphs: list[str], expected: bool):
        print()
        i = 0
        l = len(graphs)
        for graph in graphs:
            i += 1
            print(f"Testing {i}/{l} graph for UDG: {graph}")
            nxg = Graph6Converter.edge_list_to_graph(graph)
            g = Graph(nxg)
            self.assertEqual(expected, g.udg_recognition(), graph)

if __name__ == '__main__':
    unittest.main()
