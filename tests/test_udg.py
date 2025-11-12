import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import unittest
from udg import Graph
import Graph6Converter

class TestUDG(unittest.TestCase):
    
    def test_calculate_order_degree_level_desc_1(self):
        g = self._udg_graph('5: 1,2 1,3 1,4 1,5 2,3')
        g.calculate_order_degree_level(desc=True)
        self.assertEqual([0,1,2,3,4], g.order)
        
    def test_calculate_order_degree_level_asc_1(self):
        g = self._udg_graph('5: 1,2 1,3 1,4 1,5 2,3')
        g.calculate_order_degree_level(desc=False)
        self.assertEqual([3,0,4,1,2], g.order)
        
    def test_calculate_order_degree_level_desc_2(self):
        g = self._udg_graph('5: 3,2 3,1 3,4 3,5 2,1')
        g.calculate_order_degree_level(desc=True)
        self.assertEqual([2,0,1,3,4], g.order)
        
    def test_calculate_order_degree_level_asc_2(self):
        g = self._udg_graph('5: 3,2 3,1 3,4 3,5 2,1')
        g.calculate_order_degree_level(desc=False)
        self.assertEqual([3,2,4,0,1], g.order)
        
    def test_calculate_order_degree_level_desc_3(self):
        g = self._udg_graph('7: 3,1 3,2 3,4 7,5 7,6 7,2')
        g.calculate_order_degree_level(desc=True)
        self.assertEqual([2,1,0,3,6,4,5], g.order)
        
    def test_calculate_order_degree_level_asc_3(self):
        g = self._udg_graph('7: 3,1 3,2 3,4 7,5 7,6 7,2')
        g.calculate_order_degree_level(desc=False)
        self.assertEqual([0,2,3,1,6,4,5], g.order)


    def test_calculate_vertex_edge_distance_single_vertex(self):
        g = self._udg_graph('1:')
        g.calculate_vertex_edge_distance()
        self.assertEqual([[0]], g.vertex_edge_distance)

    def test_calculate_vertex_edge_distance_two_vertices_connected(self):
        g = self._udg_graph('2: 1,2')
        g.calculate_vertex_edge_distance()
        expected = [[0, 1],
                    [1, 0]]
        self.assertEqual(expected, g.vertex_edge_distance)

    def test_calculate_vertex_edge_distance_two_vertices_disconnected(self):
        g = self._udg_graph('2:')
        g.calculate_vertex_edge_distance()
        I = float('inf')
        expected = [[0, I],
                    [I, 0]]
        self.assertEqual(expected, g.vertex_edge_distance)

    def test_calculate_vertex_edge_distance_triangle(self):
        g = self._udg_graph('3: 1,2 2,3 1,3')
        g.calculate_vertex_edge_distance()
        expected = [[0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0]]
        self.assertEqual(expected, g.vertex_edge_distance)

    def test_calculate_vertex_edge_distance_path(self):
        g = self._udg_graph('4: 1,2 2,3 3,4')
        g.calculate_vertex_edge_distance()
        expected = [[0, 1, 2, 3],
                    [1, 0, 1, 2],
                    [2, 1, 0, 1],
                    [3, 2, 1, 0]]
        self.assertEqual(expected, g.vertex_edge_distance)

    def test_calculate_vertex_edge_distance_star(self):
        g = self._udg_graph('4: 1,2 1,3 1,4')
        g.calculate_vertex_edge_distance()
        expected = [[0, 1, 1, 1],
                    [1, 0, 2, 2],
                    [1, 2, 0, 2],
                    [1, 2, 2, 0]]
        self.assertEqual(expected, g.vertex_edge_distance)

    def test_calculate_vertex_edge_distance_disconnected_components(self):
        g = self._udg_graph('4: 1,2 3,4')
        g.calculate_vertex_edge_distance()
        I = float('inf')
        expected = [[0, 1, I, I],
                    [1, 0, I, I],
                    [I, I, 0, 1],
                    [I, I, 1, 0]]
        self.assertEqual(expected, g.vertex_edge_distance)

    def test_apply_order_default_dd(self):
        g = self._udg_graph('5: 1,2 1,3 1,4 1,5 2,3')
        g.apply_order()
        self.assertEqual('DD', g.order_mode) # default
        self.assertEqual([0,1,2,3,4], g.order)

    def test_apply_order_dd_explicit(self):
        g = self._udg_graph('5: 1,2 1,3 1,4 1,5 2,3')
        g.apply_order('DD')
        self.assertEqual('DD', g.order_mode)
        self.assertEqual([0,1,2,3,4], g.order)

    def test_apply_order_da(self):
        g = self._udg_graph('5: 1,2 1,3 1,4 1,5 2,3')
        g.apply_order('DA')
        self.assertEqual('DA', g.order_mode)
        self.assertEqual([3,0,4,1,2], g.order)

    def test_apply_order_custom_o(self):
        g = self._udg_graph('5: 1,2 1,3 1,4 1,5 2,3')
        g.apply_order('O_4,3,2,1,0')
        self.assertEqual('O_4,3,2,1,0', g.order_mode)
        self.assertEqual([4,3,2,1,0], g.order)

    def test_apply_order_custom_o_partial(self):
        g = self._udg_graph('5: 1,2 1,3 1,4 1,5 2,3')
        g.apply_order('O_2,4')
        self.assertEqual('O_2,4', g.order_mode)
        self.assertEqual([2,4,-1,-1,-1], g.order)

    def test_apply_order_custom_o_invalid_nodes(self):
        g = self._udg_graph('3: 1,2 2,3')
        g.apply_order('O_0,5,1')  # node 5 doesn't exist
        self.assertEqual([0,-1,1], g.order)

    def test_apply_order_force_nodes_a(self):
        g = self._udg_graph('5: 1,2 1,3 1,4 1,5 2,3')
        g.apply_order('DD', force_nodes=[3,1])
        self.assertEqual('DD', g.order_mode)
        self.assertEqual([3,1,0,2,4], g.order)

    def test_apply_order_force_nodes_b(self):
        g = self._udg_graph('4: 1,2 2,3 3,4')
        g.apply_order('DD', force_nodes=[2,1])
        self.assertEqual('DD', g.order_mode)
        self.assertEqual([2,1,0,3], g.order)

    def test_apply_order_auto_mode(self):
        g = self._udg_graph('5: 1,2 1,3 1,4 1,5 2,3')
        g.apply_order('DD_auto')
        # Should keep first two nodes and set rest to -1
        self.assertEqual('DD_auto', g.order_mode)
        self.assertEqual([0,1,-1,-1,-1], g.order)

    def test_apply_order_auto_mode_with_force_nodes(self):
        g = self._udg_graph('5: 1,2 1,3 1,4 1,5 2,3')
        g.apply_order('DD_auto', force_nodes=[4,2])
        # Should start with forced nodes then -1 for rest
        self.assertEqual('DD_auto', g.order_mode)
        self.assertEqual([4,2,-1,-1,-1], g.order)

    def test_apply_order_path_mode_a(self):
        g = self._udg_graph('4: 1,2 2,3 3,4')  # Simple path
        g.apply_order('P')
        self.assertEqual('P', g.order_mode)
        # Should find a P3 induced subgraph
        self.assertEqual([3,2,1,0], g.order)

    def test_apply_order_path_mode_b(self):
        g = self._udg_graph('6: 1,2 2,3 2,4 3,5 4,6')
        g.apply_order('P')
        self.assertEqual('P', g.order_mode)
        self.assertEqual([5,3,1,0,2,4], g.order)
        
    def test_apply_order_same_mode(self):
        g = self._udg_graph('4: 1,2 2,3 1,4')
        g.apply_order('SAME')
        self.assertEqual([0,1,2,3], g.order)

    def test_apply_order_unknown_mode_defaults_to_same(self):
        g = self._udg_graph('3: 0,1 1,2')
        g.apply_order('UNKNOWN_MODE')
        self.assertEqual('UNKNOWN_MODE', g.order_mode)
        self.assertEqual([0,1,2], g.order)
        
    def _udg_graph(self, graph: str):
        nxg = Graph6Converter.edge_list_to_graph(graph)
        g = Graph(nxg)
        return g

if __name__ == '__main__':
    unittest.main()
