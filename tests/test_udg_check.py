import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import unittest
from udg import Graph
import Graph6Converter

class TestUDG(unittest.TestCase):
    
    def test_udg_3_P_3(self):
        """BW"""
        self._check_udg(['3: 1,3 ; 2,3'], True)
        
    def test_udg_3_K_3(self):
        """Bw"""
        self._check_udg(['3: 1,2 ; 1,3 ; 2,3'], True)
    
    def test_udg_4_S_3(self):
        """CF"""
        self._check_udg(['4: 1,4 ; 2,4 ; 3,4'], True)

    def test_udg_4_P_4(self):
        """CU"""
        self._check_udg(['4: 1,3 ; 1,4 ; 2,4'], True)
        
    def test_udg_4_ga(self):
        """CV"""
        self._check_udg(['4: 1,3 ; 1,4 ; 2,4 ; 3,4'], True)
        
    def test_udg_4_C_4(self):
        """C]"""
        self._check_udg(['4: 1,3 ; 1,4 ; 2,3 ; 2,4'], True)
        
    def test_udg_4_K_4_m(self):
        """C^"""
        self._check_udg(['4: 1,3 ; 1,4 ; 2,3 ; 2,4 ; 3,4'], True)
        
    def test_udg_4_K_4(self):
        """C~"""
        self._check_udg(['4: 1,2 ; 1,3 ; 1,4 ; 2,3 ; 2,4 ; 3,4'], True)
        
    def test_udg_5_S_4(self):
        """D?{"""
        self._check_udg(['5: 1,5 ; 2,5 ; 3,5 ; 4,5'], True)
    
    def test_udg_5_ga(self):
        """DCw"""
        self._check_udg(['5: 1,4 ; 1,5 ; 2,5 ; 3,5'], True)
    
    def test_udg_5_gb(self):
        """DC{"""
        self._check_udg(['5: 1,4 ; 1,5 ; 2,5 ; 3,5 ; 4,5'], True)
    
    def test_udg_5_gc(self):
        """DEk"""
        self._check_udg(['5: 1,4 ; 1,5 ; 2,4 ; 3,5 ; 4,5'], True)
    
    def test_udg_5_gd(self):
        """DEw"""
        self._check_udg(['5: 1,4 ; 1,5 ; 2,4 ; 2,5 ; 3,5'], True)
    
    def test_udg_5_ge(self):
        """DE{"""
        self._check_udg(['5: 1,4 ; 1,5 ; 2,4 ; 2,5 ; 3,5 ; 4,5'], True)
    
    def test_udg_5_gf(self):
        """DF{"""
        self._check_udg(['5: 1,4 ; 1,5 ; 2,4 ; 2,5 ; 3,4 ; 3,5 ; 4,5'], True)
    
    def test_udg_5_P_5(self):
        """DQo"""
        self._check_udg(['5: 1,3 ; 1,5 ; 2,4 ; 2,5'], True)
    
    def test_udg_5_gg(self):
        """DQw"""
        self._check_udg(['5: 1,3 ; 1,5 ; 2,4 ; 2,5 ; 3,5'], True)
    
    def test_udg_5_gh(self):
        """DQ{"""
        self._check_udg(['5: 1,3 ; 1,5 ; 2,4 ; 2,5 ; 3,5 ; 4,5'], True)
    
    def test_udg_5_gi(self):
        """DTw"""
        self._check_udg(['5: 1,3 ; 1,4 ; 1,5 ; 2,5 ; 3,4 ; 3,5'], True)
    
    def test_udg_5_gj(self):
        """DT{"""
        self._check_udg(['5: 1,3 ; 1,4 ; 1,5 ; 2,5 ; 3,4 ; 3,5 ; 4,5'], True)
    
    def test_udg_5_C_5(self):
        """DUW"""
        self._check_udg(['5: 1,3 ; 1,4 ; 2,4 ; 2,5 ; 3,5'], True)
    
    def test_udg_5_gk(self):
        """DUw"""
        self._check_udg(['5: 1,3 ; 1,4 ; 1,5 ; 2,4 ; 2,5 ; 3,5'], True)
    
    def test_udg_5_gl(self):
        """DU{"""
        self._check_udg(['5: 1,3 ; 1,4 ; 1,5 ; 2,4 ; 2,5 ; 3,5 ; 4,5'], True)
    
    def test_udg_5_gm(self):
        """DV{"""
        self._check_udg(['5: 1,3 ; 1,4 ; 1,5 ; 2,4 ; 2,5 ; 3,4 ; 3,5 ; 4,5'], True)
    
    def test_udg_5_W_4(self):
        """D]w"""
        self._check_udg(['5: 1,3 ; 1,4 ; 1,5 ; 2,3 ; 2,4 ; 2,5 ; 3,5'], True)
    
    def test_udg_5_gn(self):
        """D]{"""
        self._check_udg(['5: 1,3 ; 1,4 ; 1,5 ; 2,3 ; 2,4 ; 2,5 ; 3,5 ; 4,5'], True)
    
    def test_udg_5_K_5_m(self):
        """D^{"""
        self._check_udg(['5: 1,3 ; 1,4 ; 1,5 ; 2,3 ; 2,4 ; 2,5 ; 3,4 ; 3,5 ; 4,5'], True)
    
    def test_udg_5_K_5(self):
        """D~{"""
        self._check_udg(['5: 1,2 ; 1,3 ; 1,4 ; 1,5 ; 2,3 ; 2,4 ; 2,5 ; 3,4 ; 3,5 ; 4,5'], True)

    @unittest.skip("Test disabled")
    def test_non_udg_5_K_2_3(self):
        """DFw"""
        self._check_udg(['5: 1,4 ; 1,5 ; 2,4 ; 2,5 ; 3,4 ; 3,5'], False)
    
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
            g.set_unit(8)
            self.assertEqual(expected, g.udg_recognition(), graph)

if __name__ == '__main__':
    unittest.main()
