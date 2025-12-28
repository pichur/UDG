import unittest
import os, sys
import textwrap
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from discrete_disk import DiscreteDisk, Coordinate, MODE_I, MODE_B, MODE_O, DISK_INNER, DISK_OUTER, create_area_by_join
import discrete_disk

TEST_SHOW = np.array(['-', '=', '+'])

class TestDiscreteDiskSqCenter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up sq_center mode before running tests."""
        cls.original_mode = discrete_disk.opts.mode
        discrete_disk.set_mode('sq_center')

    @classmethod
    def tearDownClass(cls):
        """Restore original mode after running tests."""
        discrete_disk.set_mode(cls.original_mode)

    def assertDDDEq(self, d : DiscreteDisk, content: str) :
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent(content).lstrip("\n ").rstrip("\n "))

    def test_disk_position(self):
        r = 1
        d = DiscreteDisk.disk(radius=r, x=0, y=0)
        self.assertEqual(d.x, -r)
        self.assertEqual(d.y, -r)
        self.assertEqual(d.data.shape, (2 * r + 1, 2 * r + 1))

    def test_shift_changes_only_position(self):
        r = 1
        d = DiscreteDisk.disk(radius=r, x=0, y=0)
        d.shift(2, -1)
        self.assertEqual(d.x,  2 - r)
        self.assertEqual(d.y, -1 - r)

    def test_content_c1(self):
        d = DiscreteDisk.disk(radius=1, connected=1)
        self.assertDDDEq(d, """
        ===
        =+=
        ===
        """)

    def test_content_c2(self):
        d = DiscreteDisk.disk(radius=2, connected=1)
        self.assertDDDEq(d, """
        -===-
        ==+==
        =+++=
        ==+==
        -===-
        """)

    def test_content_c3(self):
        d = DiscreteDisk.disk(radius=3, x=0, y=0, connected=1)
        self.assertDDDEq(d, """
        -=====-
        ==+++==
        =+++++=
        =+++++=
        =+++++=
        ==+++==
        -=====-
        """)

    def test_content_d3(self):
        d = DiscreteDisk.disk(radius=3, x=0, y=0, connected=0)
        self.assertDDDEq(d, """
        +=====+
        ==---==
        =-----=
        =-----=
        =-----=
        ==---==
        +=====+
        """)

    def test_content_c4(self):
        d = DiscreteDisk.disk(radius=4, connected=1)
        self.assertDDDEq(d, """
        --=====--
        -==+++==-
        ==+++++==
        =+++++++=
        =+++++++=
        =+++++++=
        ==+++++==
        -==+++==-
        --=====--
        """)

    def test_content_d4(self):
        d = DiscreteDisk.disk(radius=4, x=0, y=0, connected=0)
        self.assertDDDEq(d, """
        ++=====++
        +==---==+
        ==-----==
        =-------=
        =-------=
        =-------=
        ==-----==
        +==---==+
        ++=====++
        """)

    def test_content_c5(self):
        d = DiscreteDisk.disk(radius=5, connected=1)
        self.assertDDDEq(d, """
        ---=====---
        -===+++===-
        -=+++++++=-
        ==+++++++==
        =+++++++++=
        =+++++++++=
        =+++++++++=
        ==+++++++==
        -=+++++++=-
        -===+++===-
        ---=====---
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
        d = DiscreteDisk.disk(4)
        r = 4
        d.connect(4, 2, 1)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertDDDEq(d, """
        ---====--
        --==++==-
        --=++++==
        --=+++++=
        --=+++++=
        --==++++=
        ---==++==
        ----====-
        ---------
        """)

    def test_connect_d(self):
        d = DiscreteDisk.disk(3)
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
        d = DiscreteDisk.disk(5)
        d.disconnect(3, 2, 1)
        self.assertEqual(d.data.shape, (11, 11))
        self.assertDDDEq(d, """
        ---=====---
        -===+=====-
        -=++==---=-
        ==++=-----=
        =+++=-----=
        =+++=-----=
        =+++==---==
        ==+++======
        -=+++++++=-
        -===+++===-
        ---=====---
        """)

    def test_disconnect_out_of_range_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, -9, 9)
        self.assertEqual(d.is_all_points_O(), True)
    
    def test_create_area_by_join_CC_out_of_range(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 3, x = -8, y =  9, connected = True),
            b = DiscreteDisk.disk(radius = 3, x =  7, y = -5, connected = True))
        self.assertIs(d, DISK_OUTER)

if __name__ == '__main__':
    unittest.main()
