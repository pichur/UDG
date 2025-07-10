import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from discrete_disk import discrete_disk, shift, meet, I, B

class TestDiscreteDisk(unittest.TestCase):
    def test_disk_position(self):
        d = discrete_disk(1)
        r = 1 + 3
        self.assertEqual(d.pos, (-r, -r))
        self.assertEqual(d.data.shape, (2 * r, 2 * r))

    def test_shift_changes_only_position(self):
        d = discrete_disk(1)
        shifted = shift(d, 2, -1)
        self.assertEqual(shifted.pos, (d.pos[0] + 2, d.pos[1] - 1))
        self.assertIs(shifted.data, d.data)

    def test_meet_respects_position(self):
        a = discrete_disk(1)
        res = meet(a, a, (1, 0))
        self.assertEqual(res.pos, (-4, -4))
        self.assertEqual(res.data.shape, (8, 9))
        x, y = 0, 0
        row = y - res.pos[1]
        col = x - res.pos[0]
        self.assertEqual(res.data[row, col], I)

    def test_iter_points_order(self):
        d = discrete_disk(0)
        expected = [
            (-2, -2), (-1, -2), (0, -2), (1, -2),
            (-2, -1), (-1, -1), (0, -1), (1, -1),
            (-2, 0), (-1, 0), (0, 0), (1, 0),
            (-2, 1), (-1, 1), (0, 1), (1, 1)
        ]
        self.assertEqual(list(d.iter_points()), expected)

if __name__ == '__main__':
    unittest.main()
