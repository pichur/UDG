import unittest
import os, sys
import textwrap
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from discrete_disk import DiscreteDisk, I, B, O

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

    def test_iter_points_order(self):
        d = DiscreteDisk.disk(3)
        expected = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0), (0,  0), (1,  0),
            (-1,  1), (0,  1), (1,  1)
        ]
        self.assertEqual(d.points_list((I,)), expected)

if __name__ == '__main__':
    unittest.main()
