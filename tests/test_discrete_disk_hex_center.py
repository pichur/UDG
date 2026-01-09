import unittest
import os, sys
import textwrap
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from discrete_disk import DiscreteDisk, Coordinate, MODE_I, MODE_B, MODE_O, MODE_X,DISK_INNER, DISK_OUTER, create_area_by_join
import discrete_disk

TEST_SHOW = np.array(['-', '=', '+', '?', '.'])

class TestDiscreteDiskHexCenter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up sq_center mode before running tests."""
        cls.original_disk_mode = discrete_disk.opts.disk_mode
        discrete_disk.set_disk_mode('hex_center')

    @classmethod
    def tearDownClass(cls):
        """Restore original mode after running tests."""
        discrete_disk.set_disk_mode(cls.original_disk_mode)

    def assertDDDEq(self, d : DiscreteDisk, content: str) :
        actual = d.show(TEST_SHOW)
        expected = textwrap.dedent(content).lstrip("\n ").rstrip("\n ")
        # Compare line by line for better debugging
        actual_lines = actual.split('\n')
        expected_lines = expected.split('\n')
        max_lines = max(len(actual_lines), len(expected_lines))
        
        for i in range(max_lines):
            full_exp_line = expected_lines[i] if i < len(expected_lines) else "<missing>"
            if i < len(expected_lines):
                
                exp_line = ''.join(c for c in expected_lines[i] if c in TEST_SHOW)
            else:
                exp_line = "<missing>"
            act_line = actual_lines[i] if i < len(actual_lines) else "<missing>"

            self.assertEqual(act_line, exp_line, f"Line {i}: expected '{full_exp_line}' but got '{act_line}'")

    def test_disk_position(self):
        r = 1
        d = DiscreteDisk.disk(radius=r, x=0, y=0)
        self.assertEqual(d.x, -r)
        self.assertEqual(d.y, -2*r)
        self.assertEqual(d.data.shape, (4 * r + 1, 2 * r + 1))

    def test_shift_changes_only_position(self):
        r = 1
        d = DiscreteDisk.disk(radius=r, x=0, y=0)
        d.shift(2, -1)
        self.assertEqual(d.x,  2 - r)
        self.assertEqual(d.y, -1 - 2 * r)

    def test_content_c1(self):
        d = DiscreteDisk.disk(radius=1, connected=1)
        self.assertEqual(d.data.shape, (5, 3))
        self.assertEqual(d.x, -1)
        self.assertEqual(d.y, -2)
        self.assertDDDEq(d, """
        .=.
        =.=
        .+.
        =.=
        .=.
        """)

    def test_content_c2(self):
        d = DiscreteDisk.disk(radius=2, connected=1)
        self.assertEqual(d.data.shape, (9, 5))
        self.assertEqual(d.x, -2)
        self.assertEqual(d.y, -4)
        self.assertDDDEq(d, """
        -.=.-
        .=.=.
        =.+.=
        .+.+.
        =.+.=
        .+.+.
        =.+.=
        .=.=.
        -.=.-
        """)

    def test_content_c3(self):
        d = DiscreteDisk.disk(radius=3, x=0, y=0, connected=1)
        self.assertEqual(d.data.shape, (13, 9))
        self.assertEqual(d.x, -4)
        self.assertEqual(d.y, -6)
        self.assertDDDEq(d, """
        -.=.=.=.-
        .-.=.=.-.
        -.=.+.=.-
        .=.+.+.=.
        -.+.+.+.-
        .=.+.+.=.
        =.+.+.+.=
        .=.+.+.=.
        -.+.+.+.-
        .=.+.+.=.
        -.=.+.=.-
        .-.=.=.-.
        -.=.=.=.-
        """)

    def test_content_d3(self):
        d = DiscreteDisk.disk(radius=3, x=0, y=0, connected=0)
        self.assertEqual(d.data.shape, (13, 9))
        self.assertEqual(d.x, -4)
        self.assertEqual(d.y, -6)
        self.assertDDDEq(d, """
        +.=.=.=.+
        .+.=.=.+.
        +.=.-.=.+
        .=.-.-.=.
        +.-.-.-.+
        .=.-.-.=.
        =.-.-.-.=
        .=.-.-.=.
        +.-.-.-.+
        .=.-.-.=.
        +.=.-.=.+
        .+.=.=.+.
        +.=.=.=.+
        """)

    def test_content_c4(self):
        d = DiscreteDisk.disk(radius=4, connected=1)
        self.assertEqual(d.data.shape, (17, 11))
        self.assertEqual(d.x, -5)
        self.assertEqual(d.y, -8)
        self.assertDDDEq(d, """
        .-.=.=.=.-.
        -.=.=.=.=.-
        .-.=.+.=.-.
        -.=.+.+.=.-
        .=.+.+.+.=.
        -.+.+.+.+.-
        .=.+.+.+.=.
        =.+.+.+.+.=
        .=.+.+.+.=.
        =.+.+.+.+.=
        .=.+.+.+.=.
        -.+.+.+.+.-
        .=.+.+.+.=.
        -.=.+.+.=.-
        .-.=.+.=.-.
        -.=.=.=.=.-
        .-.=.=.=.-.
        """)

    def test_content_d4(self):
        d = DiscreteDisk.disk(radius=4, x=0, y=0, connected=0)
        self.assertEqual(d.data.shape, (17, 11))
        self.assertEqual(d.x, -5)
        self.assertEqual(d.y, -8)
        self.assertDDDEq(d, """
        .+.=.=.=.+.
        +.=.=.=.=.+
        .+.=.-.=.+.
        +.=.-.-.=.+
        .=.-.-.-.=.
        +.-.-.-.-.+
        .=.-.-.-.=.
        =.-.-.-.-.=
        .=.-.-.-.=.
        =.-.-.-.-.=
        .=.-.-.-.=.
        +.-.-.-.-.+
        .=.-.-.-.=.
        +.=.-.-.=.+
        .+.=.-.=.+.
        +.=.=.=.=.+
        .+.=.=.=.+.
        """)

    def test_content_c5(self):
        d = DiscreteDisk.disk(radius=5, connected=1)
        self.assertEqual(d.data.shape, (21, 13))
        self.assertEqual(d.x, -6)
        self.assertEqual(d.y, -10)
        self.assertDDDEq(d, """
                         10 -.-.=.=.=.-.-
                          9 .-.=.=.=.=.-.
                          8 -.=.+.+.+.=.-
                          7 .-.+.+.+.+.-.
                          6 -.=.+.+.+.=.-
                          5 .=.+.+.+.+.=.
                          4 -.+.+.+.+.+.-
                          3 .=.+.+.+.+.=.
                          2 =.+.+.+.+.+.=
                          1 .+.+.+.+.+.+.
                          0 =.+.+.+.+.+.=
                          1 .+.+.+.+.+.+.
                          2 =.+.+.+.+.+.=
                          3 .=.+.+.+.+.=.
                          4 -.+.+.+.+.+.-
                          5 .=.+.+.+.+.=.
                          6 -.=.+.+.+.=.-
                          7 .-.+.+.+.+.-.
                          8 -.=.+.+.+.=.-
                          9 .-.=.=.=.=.-.
                         10 -.-.=.=.=.-.-
                         """)

    def test_content_c6(self):
        d = DiscreteDisk.disk(radius=6, connected=1)
        self.assertEqual(d.data.shape, (25, 15))
        self.assertEqual(d.x, -7)
        self.assertEqual(d.y, -12)
        self.assertDDDEq(d, """
                         12 .-.-.=.=.=.-.-.
                         11 -.-.=.=.=.=.-.-
                         10 .-.=.+.+.+.=.-.
                          9 -.=.+.+.+.+.=.-
                          8 .-.+.+.+.+.+.-.
                          7 -.=.+.+.+.+.=.-
                          6 .=.+.+.+.+.+.=.
                          5 -.+.+.+.+.+.+.-
                          4 .=.+.+.+.+.+.=.
                          3 =.+.+.+.+.+.+.=
                          2 .+.+.+.+.+.+.+.
                          1 =.+.+.+.+.+.+.=
                          0 .+.+.+.+.+.+.+.
                          1 =.+.+.+.+.+.+.=
                          2 .+.+.+.+.+.+.+.
                          3 =.+.+.+.+.+.+.=
                          4 .=.+.+.+.+.+.=.
                          5 -.+.+.+.+.+.+.-
                          6 .=.+.+.+.+.+.=.
                          7 -.=.+.+.+.+.=.-
                          8 .-.+.+.+.+.+.-.
                          9 -.=.+.+.+.+.=.-
                         10 .-.=.+.+.+.=.-.
                         11 -.-.=.=.=.=.-.-
                         12 .-.-.=.=.=.-.-.
                         """)
        
    def test_crop_O(self):
        discrete_disk.opts.crop = True
        d = DiscreteDisk(
            data=np.array([
                [MODE_X,MODE_O,MODE_X,MODE_O,MODE_X,MODE_O,MODE_X],
                [MODE_O,MODE_X,MODE_O,MODE_X,MODE_O,MODE_X,MODE_O],
                [MODE_X,MODE_B,MODE_X,MODE_I,MODE_X,MODE_O,MODE_X],
                [MODE_I,MODE_X,MODE_B,MODE_X,MODE_O,MODE_X,MODE_O],
                [MODE_X,MODE_B,MODE_X,MODE_I,MODE_X,MODE_O,MODE_X],
                [MODE_O,MODE_X,MODE_O,MODE_X,MODE_O,MODE_X,MODE_O],
                ],
                dtype=np.uint8),
            rest=MODE_O, x = 1, y = 2).crop()
        discrete_disk.opts.crop = False
        self.assertEqual(d.x, 1) 
        self.assertEqual(d.y, 4)
        self.assertEqual(d.data.shape, (3, 4))
        self.assertEqual(d.rest, MODE_O)
        self.assertDDDEq(d, """
        .=.+
        +.=.
        .=.+
        """)

    def test_crop_I(self):
        discrete_disk.opts.crop = True
        d = DiscreteDisk(
            data=np.array([
                [MODE_I,MODE_X,MODE_O,MODE_X,MODE_O,MODE_X],
                [MODE_X,MODE_I,MODE_X,MODE_I,MODE_X,MODE_O],
                [MODE_I,MODE_X,MODE_I,MODE_X,MODE_I,MODE_X],
                [MODE_X,MODE_I,MODE_X,MODE_I,MODE_X,MODE_O],
                [MODE_I,MODE_X,MODE_I,MODE_X,MODE_I,MODE_X],
                [MODE_X,MODE_I,MODE_X,MODE_I,MODE_X,MODE_I],
                [MODE_I,MODE_X,MODE_I,MODE_X,MODE_I,MODE_X],
                ],
                dtype=np.uint8),
            rest=MODE_I, x = 1, y = 2).crop()
        discrete_disk.opts.crop = False
        self.assertEqual(d.x, 3) 
        self.assertEqual(d.y, 2)
        self.assertEqual(d.data.shape, (4, 4))
        self.assertEqual(d.rest, MODE_I)
        self.assertDDDEq(d, """
        .+.-
        +.+.
        .+.-
        -.-.
        """)

    def test_connect_a(self):
        d = DiscreteDisk.disk(3)
        d.connect(3, 1, -3)
        d.normalize()
        self.assertEqual(d.data.shape, (13, 9))
        self.assertDDDEq(d, """
                         6 -.-.-.-.-
                         5 .-.-.-.-.
                         4 -.-.-.-.-
                         3 .-.=.=.=.
                         2 -.-.=.=.-
                         1 .-.=.+.=.
                         0 -.=.+.+.=
                         1 .-.+.+.=.
                         2 -.=.+.+.-
                         3 .=.+.+.=.
                         4 -.=.+.=.-
                         5 .-.=.=.-.
                         6 -.=.=.=.-
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
        d = DiscreteDisk.disk(radius=6, connected=1)
        d.disconnect(3, 3, 1)
        self.assertEqual(d.data.shape, (25, 15))
        self.assertEqual(d.x, -7)
        self.assertEqual(d.y, -12)
        self.assertDDDEq(d, """
						 12 .-.-.=.=.=.-.-.
                         11 -.-.=.=.=.=.-.-
                         10 .-.=.+.+.+.=.-.
                          9 -.=.+.+.+.+.=.-
                          8 .-.+.+.+.+.+.-.
                          7 -.=.+.+.=.=.=.-
                          6 .=.+.+.+.=.=.=.
                          5 -.+.+.+.=.-.=.-
                          4 .=.+.+.=.-.-.=.
                          3 =.+.+.+.-.-.-.=
                          2 .+.+.+.=.-.-.=.
                          1 =.+.+.=.-.-.-.=
                          0 .+.+.+.=.-.-.=.
                          1 =.+.+.+.-.-.-.=
                          2 .+.+.+.=.-.-.=.
                          3 =.+.+.+.=.-.=.=
                          4 .=.+.+.+.=.=.=.
                          5 -.+.+.+.=.=.=.-
                          6 .=.+.+.+.+.+.=.
                          7 -.=.+.+.+.+.=.-
                          8 .-.+.+.+.+.+.-.
                          9 -.=.+.+.+.+.=.-
                         10 .-.=.+.+.+.=.-.
                         11 -.-.=.=.=.=.-.-
                         12 .-.-.=.=.=.-.-.
                         """)

    def test_disconnect_out_of_range_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, -9, 9)
        self.assertEqual(d.is_all_points_O(), True)
    
    def test_create_area_by_join_CC_out_of_range(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 3, x = -8, y =  6, connected = True),
            b = DiscreteDisk.disk(radius = 3, x =  7, y = -5, connected = True))
        self.assertIs(d, DISK_OUTER)

    def test_iter_points_default(self):
        d = DiscreteDisk.disk(radius=2, x=-1, y=3)
        p = d.points_list()
        """
            32101
            21012
        1 4 -.=.-
        0 3 .=.=.
        1 2 =.+.=
        2 1 .+.+.
        3 0 =.+.=
        4 1 .+.+.
        5 2 =.+.=
        6 3 .=.=.
        7 4 -.=.-
        """
        expected = [
                                                                    Coordinate(-1, -1, MODE_B),
                                        Coordinate(-2,  0, MODE_B),                             Coordinate(0,  0, MODE_B),
            Coordinate(-3,  1, MODE_B),                             Coordinate(-1,  1, MODE_I),                            Coordinate(1,  1, MODE_B),
                                        Coordinate(-2,  2, MODE_I),                             Coordinate(0,  2, MODE_I),
            Coordinate(-3,  3, MODE_B),                             Coordinate(-1,  3, MODE_I),                            Coordinate(1,  3, MODE_B),
                                        Coordinate(-2,  4, MODE_I),                             Coordinate(0,  4, MODE_I),
            Coordinate(-3,  5, MODE_B),                             Coordinate(-1,  5, MODE_I),                            Coordinate(1,  5, MODE_B),
                                        Coordinate(-2,  6, MODE_B),                             Coordinate(0,  6, MODE_B),
                                                                    Coordinate(-1,  7, MODE_B)
        ]
        self.assertEqual(p, expected)

    def test_iter_points_I(self):
        d = DiscreteDisk.disk(radius=2, x=2, y=-2)
        p = d.points_list(types='I')
        """
            01234
            21012
        6 4 -.=.-
        5 3 .=.=.
        4 2 =.+.=
        3 1 .+.+.
        2 0 =.+.=
        1 1 .+.+.
        0 2 =.+.=
        1 3 .=.=.
        2 4 -.=.-
        """
        expected = [
                                       Coordinate(2, -4, MODE_I),
            Coordinate(1, -3, MODE_I),                            Coordinate(3, -3, MODE_I),
                                       Coordinate(2, -2, MODE_I),
            Coordinate(1, -1, MODE_I),                            Coordinate(3, -1, MODE_I),
                                       Coordinate(2,  0, MODE_I),
        ]
        self.assertEqual(p, expected)

if __name__ == '__main__':
    unittest.main()
