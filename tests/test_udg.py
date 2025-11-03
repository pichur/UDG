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
        
    def _udg_graph(self, graph: str):
        nxg = Graph6Converter.edge_list_to_graph(graph)
        g = Graph(nxg)
        return g

if __name__ == '__main__':
    unittest.main()
