import unittest
import os, sys
import textwrap
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from discrete_disk import DiscreteDisk, Coordinate, MODE_I, MODE_B, MODE_O

TEST_SHOW = np.array(['-', '=', '+'])

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
        self.assertEqual(d.x,  2 - r)
        self.assertEqual(d.y, -1 - r)

    def test_content(self):
        d = DiscreteDisk.disk(3)
        r = 4
        self.assertEqual(d.x, -r)
        self.assertEqual(d.y, -r)
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent("""\
        ---===---
        -=======-
        -=======-
        ===+++===
        ===+++===
        ===+++===
        -=======-
        -=======-
        ---===---
        """).rstrip("\n"))

    def test_connect_a(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, 2, 0)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent("""\
        -----=---
        ---=====-
        ---=====-
        --===+===
        --===+===
        --===+===
        ---=====-
        ---=====-
        -----=---
        """).rstrip("\n"))

    def test_connect_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, -2, -1)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent("""\
        ---------
        -===-----
        -=====---
        ======---
        ===+===--
        ===+===--
        -======--
        -=====---
        ---===---
        """).rstrip("\n"))

    def test_connect_disk_a(self):
        d = DiscreteDisk.disk(3)
        r = 4
        e = DiscreteDisk.disk(3, -2, -1)
        d.connect_disk(e)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.x, -r)
        self.assertEqual(d.y, -r)
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent("""\
        ---------
        -===-----
        -=====---
        ======---
        ===+===--
        ===+===--
        -======--
        -=====---
        ---===---
        """).rstrip("\n"))

    def test_connect_disk_b(self):
        d = DiscreteDisk.disk(3, 2, 1)
        r = 4
        e = DiscreteDisk.disk(3)
        d.connect_disk(e)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.x, 2 - r)
        self.assertEqual(d.y, 1 - r)
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent("""\
        ---------
        -===-----
        -=====---
        ======---
        ===+===--
        ===+===--
        -======--
        -=====---
        ---===---
        """).rstrip("\n"))

    def test_connect_disk_c(self):
        d = DiscreteDisk.disk(3, 4, 5)
        r = 4
        e = DiscreteDisk.disk(3, 2, 4)
        d.connect_disk(e)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.x, 4 - r)
        self.assertEqual(d.y, 5 - r)
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent("""\
        ---------
        -===-----
        -=====---
        ======---
        ===+===--
        ===+===--
        -======--
        -=====---
        ---===---
        """).rstrip("\n"))

    def test_connect_c(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, 4, 4)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent("""\
        ----==---
        ----====-
        -----===-
        -----====
        -------==
        ---------
        ---------
        ---------
        ---------
        """).rstrip("\n"))

    def test_connect_d(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, 8, 8)
        self.assertEqual(d.is_all_points_O(), True)

    def test_connect_out_of_range_a(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, 10, -2)
        self.assertEqual(d.is_all_points_O(), True)

    def test_connect_out_of_range_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, -9, 9)
        self.assertEqual(d.is_all_points_O(), True)

    def test_disconnect_a(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.disconnect(3, 2, 0)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent("""\
        ---===---
        -=======-
        -=======-
        =====---=
        =====---=
        =====---=
        -=======-
        -=======-
        ---===---
        """).rstrip("\n"))

    def test_disconnect_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.disconnect(3, -2, -1)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent("""\
        ---===---
        -=======-
        -=======-
        =========
        =---=====
        =---=====
        ----====-
        -=======-
        ---===---
        """).rstrip("\n"))

    def test_disconnect_c(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.disconnect(3, 4, 4)
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent("""\
        ---===---
        -======--
        -=======-
        ===++====
        ===+++===
        ===+++===
        -=======-
        -=======-
        ---===---
        """).rstrip("\n"))

    def test_disconnect_d(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.disconnect(3, 8, 8)
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent("""\
        ---===---
        -=======-
        -=======-
        ===+++===
        ===+++===
        ===+++===
        -=======-
        -=======-
        ---===---
        """).rstrip("\n"))

    def test_disconnect_out_of_range_a(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.disconnect(3, 10, -2)
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent("""\
        ---===---
        -=======-
        -=======-
        ===+++===
        ===+++===
        ===+++===
        -=======-
        -=======-
        ---===---
        """).rstrip("\n"))

    def test_disconnect_out_of_range_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, -9, 9)
        self.assertEqual(d.is_all_points_O(), True)

    def test_iter_points(self):
        d = DiscreteDisk.disk(3)
        p = d.points_list()

        # MODE_O
        self.assertNotIn(Coordinate(4, 2, MODE_O), p)
        self.assertNotIn(Coordinate(4, 2, MODE_B), p)
        self.assertNotIn(Coordinate(4, 2, MODE_I), p)

        # MODE_B
        self.assertNotIn(Coordinate(3, -3, MODE_O), p)
        self.assertIn   (Coordinate(3, -3, MODE_B), p)
        self.assertNotIn(Coordinate(3, -3, MODE_I), p)

        # MODE_I
        self.assertNotIn(Coordinate(-1, 0, MODE_O), p)
        self.assertNotIn(Coordinate(-1, 0, MODE_B), p)
        self.assertIn   (Coordinate(-1, 0, MODE_I), p)

        expected = [
            Coordinate(-1, -1, MODE_I), Coordinate(0, -1, MODE_I), Coordinate(1, -1, MODE_I),
            Coordinate(-1,  0, MODE_I), Coordinate(0,  0, MODE_I), Coordinate(1,  0, MODE_I),
            Coordinate(-1,  1, MODE_I), Coordinate(0,  1, MODE_I), Coordinate(1,  1, MODE_I)
        ]
        self.assertEqual(d.points_list((MODE_I,)), expected)

    def test_iter_points_order(self):
        d = DiscreteDisk.disk(3)
        expected = [
            Coordinate(-1, -1, MODE_I), Coordinate(0, -1, MODE_I), Coordinate(1, -1, MODE_I),
            Coordinate(-1,  0, MODE_I), Coordinate(0,  0, MODE_I), Coordinate(1,  0, MODE_I),
            Coordinate(-1,  1, MODE_I), Coordinate(0,  1, MODE_I), Coordinate(1,  1, MODE_I)
        ]
        self.assertEqual(d.points_list((MODE_I,)), expected)

if __name__ == '__main__':
    unittest.main()
