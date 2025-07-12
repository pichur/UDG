import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from discrete_disk import DiscreteDisk, I, B, O

class TestDiscreteDisk(unittest.TestCase):
    def test_disk_position(self):
        d = DiscreteDisk.disk(1)
        r = 2
        self.assertEqual(d.x, -r)
        self.assertEqual(d.y, -r)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))

    def test_shift_changes_only_position(self):
        d = DiscreteDisk.disk(1)
        r = 2
        d.shift(2, -1)
        self.assertEqual(d.x, -r + 2)
        self.assertEqual(d.y, -r - 1)

    def test_content(self):
        d = DiscreteDisk.disk(3)
        self.assertEqual(d.data[0, 0], O)
        self.assertEqual(d.data[1, 0], O)
        self.assertEqual(d.data[2, 1], B)
        self.assertEqual(d.data[3, 1], B)
        self.assertEqual(d.data[4, 2], B)
        self.assertEqual(d.data[5, 2], B)
        self.assertEqual(d.data[6, 3], B)
        self.assertEqual(d.data[5, 3], I)
        self.assertEqual(d.data[4, 4], I)
        self.assertEqual(d.data[3, 4], I)
        self.assertEqual(d.data[2, 5], B)
        self.assertEqual(d.data[1, 5], B)
        self.assertEqual(d.data[0, 6], O)

    def test_connect_a(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, 2, 0)
        self.assertEqual(d.x, -r)
        self.assertEqual(d.y, -r)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.data[1, 0], O)
        self.assertEqual(d.data[1, 1], O)
        self.assertEqual(d.data[2, 2], O)
        self.assertEqual(d.data[2, 3], B)
        self.assertEqual(d.data[3, 4], B)
        self.assertEqual(d.data[3, 5], I)
        self.assertEqual(d.data[4, 6], B)
        self.assertEqual(d.data[4, 7], B)
        self.assertEqual(d.data[5, 8], B)

    def test_connect_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, -2, -1)
        self.assertEqual(d.x, -r)
        self.assertEqual(d.y, -r)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.data[1, 0], O)
        self.assertEqual(d.data[1, 1], O)
        self.assertEqual(d.data[2, 2], O)
        self.assertEqual(d.data[2, 3], B)
        self.assertEqual(d.data[3, 3], B)
        self.assertEqual(d.data[3, 4], I)
        self.assertEqual(d.data[4, 5], B)
        self.assertEqual(d.data[4, 6], B)
        self.assertEqual(d.data[5, 7], B)
        self.assertEqual(d.data[5, 8], B)

    def test_connect_out_of_range(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, 5, -1)
        self.assertEqual(d.x, -r)
        self.assertEqual(d.y, -r)
        self.assertEqual(d.is_all_points_O(), True)

    def test_iter_points_order(self):
        d = DiscreteDisk.disk(3)
        expected = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0), (0,  0), (1,  0),
            (-1,  1), (0,  1), (1,  1)
        ]
        self.assertEqual(list(d.iter_points(I)), expected)

if __name__ == '__main__':
    unittest.main()
