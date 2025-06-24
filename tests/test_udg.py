import unittest
from udg import Graph, udg_recognition

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

if __name__ == '__main__':
    unittest.main()
