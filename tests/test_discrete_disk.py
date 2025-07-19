import unittest
import os, sys
import textwrap
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from discrete_disk import DiscreteDisk, Coordinate, MODE_I, MODE_B, MODE_O, DISK_INNER, DISK_OUTER, create_area_by_join
import discrete_disk

TEST_SHOW = np.array(['-', '=', '+'])

class TestDiscreteDisk(unittest.TestCase):

    def assertDDDEq(self, d : DiscreteDisk, content: str) :
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent(content).lstrip("\n ").rstrip("\n "))

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
        self.assertDDDEq(d, """
        ---===---
        -=======-
        -=======-
        ===+++===
        ===+++===
        ===+++===
        -=======-
        -=======-
        ---===---
        """)

    def test_crop_O(self):
        discrete_disk.opts.crop = True
        d = DiscreteDisk(
            data=np.array([
                [MODE_O,MODE_O,MODE_O],
                [MODE_O,MODE_O,MODE_O],
                [MODE_B,MODE_I,MODE_O],
                [MODE_I,MODE_B,MODE_O],
                [MODE_B,MODE_I,MODE_O],
                [MODE_O,MODE_O,MODE_O],
                ],
                dtype=np.uint8),
            rest=MODE_O, x = 1, y = 2).crop()
        discrete_disk.opts.crop = False
        self.assertEqual(d.x, 1) 
        self.assertEqual(d.y, 4)
        self.assertEqual(d.data.shape, (3, 2))
        self.assertEqual(d.rest, MODE_O)
        self.assertDDDEq(d, """
        =+
        +=
        =+
        """)

    def test_crop_I(self):
        discrete_disk.opts.crop = True
        d = DiscreteDisk(
            data=np.array([
                [MODE_I,MODE_O,MODE_O],
                [MODE_I,MODE_I,MODE_O],
                [MODE_I,MODE_I,MODE_I],
                [MODE_I,MODE_I,MODE_O],
                [MODE_I,MODE_I,MODE_I],
                [MODE_I,MODE_I,MODE_I],
                [MODE_I,MODE_I,MODE_I],
                ],
                dtype=np.uint8),
            rest=MODE_I, x = 1, y = 2).crop()
        discrete_disk.opts.crop = False
        self.assertEqual(d.x, 2) 
        self.assertEqual(d.y, 2)
        self.assertEqual(d.data.shape, (4, 2))
        self.assertEqual(d.rest, MODE_I)
        self.assertDDDEq(d, """
        +-
        ++
        +-
        --
        """)

    def test_connect_a(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, 2, 0)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertDDDEq(d, """
        -----=---
        ---=====-
        ---=====-
        --===+===
        --===+===
        --===+===
        ---=====-
        ---=====-
        -----=---
        """)

    def test_connect_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, -2, -1)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertDDDEq(d, """
        ---------
        -===-----
        -=====---
        ======---
        ===+===--
        ===+===--
        -======--
        -=====---
        ---===---
        """)

    def test_connect_disk_a(self):
        d = DiscreteDisk.disk(3)
        r = 4
        e = DiscreteDisk.disk(3, -2, -1)
        d.connect_disk(e)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.x, -r)
        self.assertEqual(d.y, -r)
        self.assertDDDEq(d, """
        ---------
        -===-----
        -=====---
        ======---
        ===+===--
        ===+===--
        -======--
        -=====---
        ---===---
        """)

    def test_connect_disk_b(self):
        d = DiscreteDisk.disk(3, 2, 1)
        r = 4
        e = DiscreteDisk.disk(3)
        d.connect_disk(e)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.x, 2 - r)
        self.assertEqual(d.y, 1 - r)
        self.assertDDDEq(d, """
        ---------
        -===-----
        -=====---
        ======---
        ===+===--
        ===+===--
        -======--
        -=====---
        ---===---
        """)

    def test_connect_disk_c(self):
        d = DiscreteDisk.disk(3, 4, 5)
        r = 4
        e = DiscreteDisk.disk(3, 2, 4)
        d.connect_disk(e)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertEqual(d.x, 4 - r)
        self.assertEqual(d.y, 5 - r)
        self.assertDDDEq(d, """
        ---------
        -===-----
        -=====---
        ======---
        ===+===--
        ===+===--
        -======--
        -=====---
        ---===---
        """)

    def test_connect_c(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, 4, 4)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertDDDEq(d, """
        ----==---
        ----====-
        -----===-
        -----====
        -------==
        ---------
        ---------
        ---------
        ---------
        """)

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
        self.assertDDDEq(d, """
        ---===---
        -=======-
        -=======-
        =====---=
        =====---=
        =====---=
        -=======-
        -=======-
        ---===---
        """)

    def test_disconnect_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.disconnect(3, -2, -1)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))
        self.assertDDDEq(d, """
        ---===---
        -=======-
        -=======-
        =========
        =---=====
        =---=====
        ----====-
        -=======-
        ---===---
        """)

    def test_disconnect_c(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.disconnect(3, 4, 4)
        self.assertDDDEq(d, """
        ---===---
        -======--
        -=======-
        ===++====
        ===+++===
        ===+++===
        -=======-
        -=======-
        ---===---
        """)

    def test_disconnect_d(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.disconnect(3, 8, 8)
        self.assertDDDEq(d, """
        ---===---
        -=======-
        -=======-
        ===+++===
        ===+++===
        ===+++===
        -=======-
        -=======-
        ---===---
        """)

    def test_disconnect_out_of_range_a(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.disconnect(3, 10, -2)
        self.assertDDDEq(d, """
        ---===---
        -=======-
        -=======-
        ===+++===
        ===+++===
        ===+++===
        -=======-
        -=======-
        ---===---
        """)

    def test_disconnect_out_of_range_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, -9, 9)
        self.assertEqual(d.is_all_points_O(), True)
    
    def test_create_area_by_join_CC(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 3, x = -7, y =  1, connected = True),
            b = DiscreteDisk.disk(radius = 3, x = -6, y = -1, connected = True))
        self.assertEqual(d.x, -10) 
        self.assertEqual(d.y, - 3)
        self.assertEqual(d.data.shape, (7, 8))
        self.assertEqual(d.rest, MODE_O)
        self.assertDDDEq(d, """
        ---===--
        -=======
        -=======
        ===++===
        =======-
        =======-
        --===---
        """)

    def test_create_area_by_join_CC_out_of_range(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 3, x = -8, y =  9, connected = True),
            b = DiscreteDisk.disk(radius = 3, x =  7, y = -5, connected = True))
        self.assertIs(d, DISK_OUTER)

    def test_create_area_by_join_CD(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 3, x = -7, y =  1, connected = True),
            b = DiscreteDisk.disk(radius = 3, x = -6, y = -1, connected = False))
        self.assertEqual(d.x, -11) 
        self.assertEqual(d.y, - 3)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertEqual(d.rest, MODE_O)
        self.assertDDDEq(d, """
        ---===---
        -=======-
        -=======-
        =========
        =========
        ====---==
        -===---=-
        -===---=-
        ---===---
        """)

    def test_create_area_by_join_DC(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 3, x = -7, y =  1, connected = False),
            b = DiscreteDisk.disk(radius = 3, x = -6, y = -1, connected = True))
        self.assertEqual(d.x, -10)
        self.assertEqual(d.y, - 5)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertEqual(d.rest, MODE_O)
        self.assertDDDEq(d, """
        ---===---
        -=---===-
        -=---===-
        ==---====
        =========
        =========
        -=======-
        -=======-
        ---===---
        """)

    def test_create_area_by_join_DC_inner(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 3, x = -7, y =  1, connected = False),
            b = DiscreteDisk.disk(radius = 3, x = -6, y = -3, connected = True))
        self.assertEqual(d.x, -10)
        self.assertEqual(d.y, - 7)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertEqual(d.rest, MODE_O)
        self.assertDDDEq(d, """
        -----=---
        -=---===-
        -=======-
        =========
        =====+===
        ===+++===
        -=======-
        -=======-
        ---===---
        """)

    def test_create_area_by_join_DC_crop(self):
        discrete_disk.opts.crop = True
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 3, x = -7, y =  1, connected = False),
            b = DiscreteDisk.disk(radius = 3, x = -7, y = -3, connected = True))
        discrete_disk.opts.crop = False
        self.assertEqual(d.x, -11)
        self.assertEqual(d.y, - 7)
        self.assertEqual(d.data.shape, (8, 9))
        self.assertEqual(d.rest, MODE_O)
        self.assertDDDEq(d, """
        -==---==-
        -=======-
        =========
        =========
        ===+++===
        -=======-
        -=======-
        ---===---
        """)

    def test_create_area_by_join_DD(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 3, x = -7, y =  1, connected = False),
            b = DiscreteDisk.disk(radius = 3, x = -6, y = -1, connected = False))
        self.assertEqual(d.x, -11)
        self.assertEqual(d.y, - 5)
        self.assertEqual(d.data.shape, (11, 10))
        self.assertEqual(d.rest, MODE_I)
        self.assertDDDEq(d, """
        +++===++++
        +=======++
        +=======++
        ===---===+
        ===---===+
        ===----===
        +===---===
        +===---===
        ++=======+
        ++=======+
        ++++===+++
        """)

    def test_create_area_by_join_DD_out_of_range(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 3, x = -8, y =  9, connected = False),
            b = DiscreteDisk.disk(radius = 3, x =  7, y = -5, connected = False))
        self.assertEqual(d.x, -12)
        self.assertEqual(d.y, - 9)
        self.assertEqual(d.data.shape, (23, 24))
        self.assertEqual(d.rest, MODE_I)
        self.assertDDDEq(d, """
        +++===++++++++++++++++++
        +=======++++++++++++++++
        +=======++++++++++++++++
        ===---===+++++++++++++++
        ===---===+++++++++++++++
        ===---===+++++++++++++++
        +=======++++++++++++++++
        +=======++++++++++++++++
        +++===++++++++++++++++++
        ++++++++++++++++++++++++
        ++++++++++++++++++++++++
        ++++++++++++++++++++++++
        ++++++++++++++++++++++++
        ++++++++++++++++++++++++
        ++++++++++++++++++===+++
        ++++++++++++++++=======+
        ++++++++++++++++=======+
        +++++++++++++++===---===
        +++++++++++++++===---===
        +++++++++++++++===---===
        ++++++++++++++++=======+
        ++++++++++++++++=======+
        ++++++++++++++++++===+++
        """)

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
        self.assertEqual(d.points_list('I'), expected)

    def test_iter_points_order(self):
        d = DiscreteDisk.disk(3)
        expected = [
            Coordinate(-1, -1, MODE_I), Coordinate(0, -1, MODE_I), Coordinate(1, -1, MODE_I),
            Coordinate(-1,  0, MODE_I), Coordinate(0,  0, MODE_I), Coordinate(1,  0, MODE_I),
            Coordinate(-1,  1, MODE_I), Coordinate(0,  1, MODE_I), Coordinate(1,  1, MODE_I)
        ]
        self.assertEqual(d.points_list('I'), expected)

if __name__ == '__main__':
    unittest.main()
