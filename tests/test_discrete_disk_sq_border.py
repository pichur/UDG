import unittest
import os, sys, time
import textwrap
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from discrete_disk import DiscreteDisk, Coordinate, MODE_I, MODE_B, MODE_O, DISK_INNER, DISK_OUTER, create_area_by_join
import discrete_disk

TEST_SHOW = np.array(['-', '=', '+'])

class TestDiscreteDiskSqBorder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up sq_center mode before running tests."""
        cls.original_disk_mode = discrete_disk.opts.disk_mode
        discrete_disk.set_disk_mode('sq_border')

    @classmethod
    def tearDownClass(cls):
        """Restore original mode after running tests."""
        discrete_disk.set_disk_mode(cls.original_disk_mode)

    def assertDDDEq(self, d : DiscreteDisk, content: str) :
        self.assertEqual(d.show(TEST_SHOW), textwrap.dedent(content).lstrip("\n ").rstrip("\n "))

    def test_disk_position(self):
        d = DiscreteDisk.disk(1)
        self.assertEqual(d.x, -1)
        self.assertEqual(d.y, -1)
        self.assertEqual(d.data.shape, (3, 3))

    def test_shift_changes_only_position(self):
        d = DiscreteDisk.disk(1)
        d.shift(2, -1)
        self.assertEqual(d.x,  1)
        self.assertEqual(d.y, -2)

    def test_content_c1(self):
        d = DiscreteDisk.disk(radius=1, connected=1)
        self.assertDDDEq(d, """
        -==
        ===
        -=-
        """)

    def test_content_c2(self):
        d = DiscreteDisk.disk(radius=2, connected=1)
        self.assertDDDEq(d, """
        -====
        -=++=
        ==++=
        -====
        --=--
        """)

    def test_content_c3(self):
        d = DiscreteDisk.disk(radius=3, connected=1)
        self.assertDDDEq(d, """
        -======
        -=++++=
        -=++++=
        ==++++=
        -=++++=
        -======
        ---=---
        """)

    def test_content_c4(self):
        d = DiscreteDisk.disk(radius=4, connected=1)
        self.assertDDDEq(d, """
        --======-
        -==++++==
        -=++++++=
        -=++++++=
        ==++++++=
        -=++++++=
        -==++++==
        --======-
        ----=----
        """)

    def test_content_d4(self):
        d = DiscreteDisk.disk(radius=4, connected=0)
        self.assertDDDEq(d, """
        ++======+
        +==----==
        +=------=
        +=------=
        ==------=
        +=------=
        +==----==
        ++======+
        ++++=++++
        """)

    def test_content_c5(self):
        d = DiscreteDisk.disk(radius=5, connected=1)
        self.assertDDDEq(d, """
        ---======--
        --=++++++=-
        -=++++++++=
        -=++++++++=
        -=++++++++=
        ==++++++++=
        -=++++++++=
        -=++++++++=
        -==++++++=-
        --=======--
        -----=-----
        """)

    def test_content_c6(self):
        d = DiscreteDisk.disk(radius=6, connected=1)
        self.assertDDDEq(d, """
        ---========--
        --==++++++==-
        -==++++++++==
        -=++++++++++=
        -=++++++++++=
        -=++++++++++=
        ==++++++++++=
        -=++++++++++=
        -=++++++++++=
        -==++++++++==
        --==++++++==-
        ---========--
        ------=------
        """)

    def test_content_c7(self):
        d = DiscreteDisk.disk(radius=7, connected=1)
        self.assertDDDEq(d, """
        ----========---
        ---==++++++==--
        --==++++++++==-
        -==++++++++++==
        -=++++++++++++=
        -=++++++++++++=
        -=++++++++++++=
        ==++++++++++++=
        -=++++++++++++=
        -=++++++++++++=
        -==++++++++++==
        --==++++++++==-
        ---==++++++==--
        ----========---
        -------=-------
        """)

    def test_content_c8(self):
        d = DiscreteDisk.disk(radius=8, connected=1)
        self.assertDDDEq(d, """
        -----========----
        ---===++++++===--
        --==++++++++++==-
        --=++++++++++++=-
        -==++++++++++++==
        -=++++++++++++++=
        -=++++++++++++++=
        -=++++++++++++++=
        ==++++++++++++++=
        -=++++++++++++++=
        -=++++++++++++++=
        -==++++++++++++==
        --=++++++++++++=-
        --==++++++++++==-
        ---===++++++===--
        -----========----
        --------=--------
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
        d.connect(4, 2, 0)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertDDDEq(d, """
        ----====-
        ---==++==
        ---=++++=
        ---=++++=
        --==++++=
        ---=++++=
        ---==++==
        ----====-
        ---------
        """)

    def test_connect_b(self):
        d = DiscreteDisk.disk(4)
        d.connect(4, -2, -1)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertDDDEq(d, """
        ---------
        -=====---
        -=+++==--
        -=++++=--
        ==++++=--
        -=++++=--
        -==+++=--
        --=====--
        ----=----
        """)

    def test_connect_c(self):
        d = DiscreteDisk.disk(4)
        d.connect(4, 3, 3)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertDDDEq(d, """
        ----====-
        ---==++==
        ----=+++=
        ----==++=
        -----====
        -------=-
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
        d = DiscreteDisk.disk(4)
        d.disconnect(4, 2, 0)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertDDDEq(d, """
        --======-
        -====----
        -=+=-----
        -=+=-----
        ====-----
        -=+=-----
        -====----
        --======-
        ----=----
        """)

    def test_disconnect_b(self):
        d = DiscreteDisk.disk(4)
        d.disconnect(4, -2, -1)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertDDDEq(d, """
        --======-
        -=====+==
        -----==+=
        ------=+=
        ------=+=
        ------=+=
        ------===
        -----===-
        ----=----
        """)

    def test_disconnect_c(self):
        d = DiscreteDisk.disk(4)
        d.disconnect(3, 4, 4)
        self.assertDDDEq(d, """
        --=====--
        -==+++=--
        -=++++===
        -=++++++=
        ==++++++=
        -=++++++=
        -==++++==
        --======-
        ----=----
        """)

    def test_disconnect_d(self):
        d = DiscreteDisk.disk(4)
        d.disconnect(3, 8, 8)
        self.assertDDDEq(d, """
        --======-
        -==++++==
        -=++++++=
        -=++++++=
        ==++++++=
        -=++++++=
        -==++++==
        --======-
        ----=----
        """)

    def test_disconnect_out_of_range_a(self):
        d = DiscreteDisk.disk(4)
        d.disconnect(3, 10, -2)
        self.assertDDDEq(d, """
        --======-
        -==++++==
        -=++++++=
        -=++++++=
        ==++++++=
        -=++++++=
        -==++++==
        --======-
        ----=----
        """)

    def test_disconnect_out_of_range_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, -9, 9)
        self.assertEqual(d.is_all_points_O(), True)
    
    def test_create_area_by_join_CC(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -7, y =  1, connected = 1),
            b = DiscreteDisk.disk(radius = 4, x = -6, y = -1, connected = 1))
        self.assertEqual(d.x, -10) 
        self.assertEqual(d.y, - 3)
        self.assertEqual(d.data.shape, (7, 8))
        self.assertEqual(d.rest, MODE_O)
        self.assertDDDEq(d, """
        --======
        -==++++=
        -=+++++=
        -=+++++=
        ==++++==
        -======-
        ---=----
        """)

    def test_create_area_by_join_CC_out_of_range(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 3, x = -8, y =  9, connected = 1),
            b = DiscreteDisk.disk(radius = 3, x =  7, y = -5, connected = 1))
        self.assertIs(d, DISK_OUTER)

    def test_create_area_by_join_CD(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -7, y =  1, connected = 1),
            b = DiscreteDisk.disk(radius = 4, x = -6, y = -1, connected = 0))
        self.assertEqual(d.x, -7 - 4) 
        self.assertEqual(d.y,  1 - 4)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertEqual(d.rest, MODE_O)
        self.assertDDDEq(d, """
        --======-
        -==++++==
        -=+======
        -===----=
        ===------
        -==------
        -==------
        --=------
        ---------
        """)

    def test_create_area_by_join_CD_crop(self):
        discrete_disk.opts.crop = True
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -7, y =  1, connected = 1),
            b = DiscreteDisk.disk(radius = 4, x = -6, y = -1, connected = 0))
        discrete_disk.opts.crop = False
        self.assertEqual(d.x, -7 - 4) 
        self.assertEqual(d.y,  1 - 4 + 1)
        self.assertEqual(d.data.shape, (8, 9))
        self.assertEqual(d.rest, MODE_O)
        self.assertDDDEq(d, """
        --======-
        -==++++==
        -=+======
        -===----=
        ===------
        -==------
        -==------
        --=------
        """)

    def test_create_area_by_join_DC(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -7, y =  1, connected = 0),
            b = DiscreteDisk.disk(radius = 4, x = -6, y = -1, connected = 1))
        self.assertEqual(d.x, -6 - 4)
        self.assertEqual(d.y, -1 - 4)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertEqual(d.rest, MODE_O)
        self.assertDDDEq(d, """
        -------=-
        -------==
        -------==
        -------==
        ==----===
        -======+=
        -===+++==
        --======-
        ----=----
        """)

    def test_create_area_by_join_DC_crop(self):
        discrete_disk.opts.crop = True
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -7, y =  1, connected = 0),
            b = DiscreteDisk.disk(radius = 4, x = -7, y = -3, connected = 1))
        discrete_disk.opts.crop = False
        self.assertEqual(d.x, -7 - 4)
        self.assertEqual(d.y, -3 - 4)
        self.assertEqual(d.data.shape, (8, 9))
        self.assertEqual(d.rest, MODE_O)
        self.assertDDDEq(d, """
        -=------=
        -==----==
        -========
        ==++=+++=
        -=++++++=
        -==++++==
        --======-
        ----=----
        """)

    def test_create_area_by_join_DD(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -7, y =  1, connected = 0),
            b = DiscreteDisk.disk(radius = 4, x = -6, y = -1, connected = 0))
        self.assertEqual(d.x, -7 - 4)
        self.assertEqual(d.y, -1 - 4)
        self.assertEqual(d.data.shape, (11, 10))
        self.assertEqual(d.rest, MODE_I)
        self.assertDDDEq(d, """
        ++======++
        +==----==+
        +=------=+
        +=------==
        ==-------=
        +=-------=
        +==------=
        ++=------=
        ++==----==
        +++======+
        +++++=++++
        """)

    def test_create_area_by_join_DD_out_of_range(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -8, y =  9, connected = 0),
            b = DiscreteDisk.disk(radius = 4, x =  3, y = -1, connected = 0))
        self.assertEqual(d.x, -12)
        self.assertEqual(d.y, - 5)
        self.assertEqual(d.data.shape, (19, 20))
        self.assertEqual(d.rest, MODE_I)
        self.assertDDDEq(d, """
        ++======++++++++++++
        +==----==+++++++++++
        +=------=+++++++++++
        +=------=+++++++++++
        ==------=+++++++++++
        +=------=+++++++++++
        +==----==+++++++++++
        ++======++++++++++++
        ++++=+++++++++++++++
        ++++++++++++++++++++
        +++++++++++++======+
        ++++++++++++==----==
        ++++++++++++=------=
        ++++++++++++=------=
        +++++++++++==------=
        ++++++++++++=------=
        ++++++++++++==----==
        +++++++++++++======+
        +++++++++++++++=++++
        """)

    def test_create_area_by_join_D3C6_0(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = 0, y = 0, connected = 0),
            b = DiscreteDisk.disk(radius = 7, x = 0, y = 0, connected = 1))
        self.assertEqual(d.x, -7)
        self.assertEqual(d.y, -7)
        self.assertEqual(d.data.shape, (15, 15))
        self.assertEqual(d.rest, MODE_O)

        self.assertDDDEq(d, """
        ----========---
        ---==++++++==--
        --==++++++++==-
        -==++======++==
        -=++==----==++=
        -=++=------=++=
        -=++=------=++=
        ==+==------=++=
        -=++=------=++=
        -=++==----==++=
        -==++======++==
        --==+++=++++==-
        ---==++++++==--
        ----========---
        -------=-------
        """)
    
    def test_iter_points(self):
        d = DiscreteDisk.disk(4)
        p = d.points_list()

        # MODE_O
        self.assertNotIn(Coordinate(4, 4, MODE_O), p)
        self.assertNotIn(Coordinate(4, 4, MODE_B), p)
        self.assertNotIn(Coordinate(4, 4, MODE_I), p)

        # MODE_B
        self.assertNotIn(Coordinate(3, -3, MODE_O), p)
        self.assertIn   (Coordinate(3, -3, MODE_B), p)
        self.assertNotIn(Coordinate(3, -3, MODE_I), p)

        # MODE_I
        self.assertNotIn(Coordinate(-1, 1, MODE_O), p)
        self.assertNotIn(Coordinate(-1, 1, MODE_B), p)
        self.assertIn   (Coordinate(-1, 1, MODE_I), p)

        expected = [
                                                                                                                            Coordinate(0, -4, MODE_B),
                                                                    Coordinate(-2, -3, MODE_B), Coordinate(-1, -3, MODE_B), Coordinate(0, -3, MODE_B), Coordinate(1, -3, MODE_B), Coordinate(2, -3, MODE_B), Coordinate(3, -3, MODE_B),
                                        Coordinate(-3, -2, MODE_B), Coordinate(-2, -2, MODE_B), Coordinate(-1, -2, MODE_I), Coordinate(0, -2, MODE_I), Coordinate(1, -2, MODE_I), Coordinate(2, -2, MODE_I), Coordinate(3, -2, MODE_B), Coordinate(4, -2, MODE_B),
                                        Coordinate(-3, -1, MODE_B), Coordinate(-2, -1, MODE_I), Coordinate(-1, -1, MODE_I), Coordinate(0, -1, MODE_I), Coordinate(1, -1, MODE_I), Coordinate(2, -1, MODE_I), Coordinate(3, -1, MODE_I), Coordinate(4, -1, MODE_B),
            Coordinate(-4,  0, MODE_B), Coordinate(-3,  0, MODE_B), Coordinate(-2,  0, MODE_I), Coordinate(-1,  0, MODE_I), Coordinate(0,  0, MODE_I), Coordinate(1,  0, MODE_I), Coordinate(2,  0, MODE_I), Coordinate(3,  0, MODE_I), Coordinate(4,  0, MODE_B),
                                        Coordinate(-3,  1, MODE_B), Coordinate(-2,  1, MODE_I), Coordinate(-1,  1, MODE_I), Coordinate(0,  1, MODE_I), Coordinate(1,  1, MODE_I), Coordinate(2,  1, MODE_I), Coordinate(3,  1, MODE_I), Coordinate(4,  1, MODE_B),
                                        Coordinate(-3,  2, MODE_B), Coordinate(-2,  2, MODE_I), Coordinate(-1,  2, MODE_I), Coordinate(0,  2, MODE_I), Coordinate(1,  2, MODE_I), Coordinate(2,  2, MODE_I), Coordinate(3,  2, MODE_I), Coordinate(4,  2, MODE_B),
                                        Coordinate(-3,  3, MODE_B), Coordinate(-2,  3, MODE_B), Coordinate(-1,  3, MODE_I), Coordinate(0,  3, MODE_I), Coordinate(1,  3, MODE_I), Coordinate(2,  3, MODE_I), Coordinate(3,  3, MODE_B), Coordinate(4,  3, MODE_B),
                                                                    Coordinate(-2,  4, MODE_B), Coordinate(-1,  4, MODE_B), Coordinate(0,  4, MODE_B), Coordinate(1,  4, MODE_B), Coordinate(2,  4, MODE_B), Coordinate(3,  4, MODE_B)
        ]
        self.assertEqual(d.points_list(), expected)

    def test_iter_points_order(self):
        d = DiscreteDisk.disk(radius=3, x=2, y=-1)
        expected = [
            Coordinate(1, -2, MODE_I), Coordinate(2, -2, MODE_I), Coordinate(3, -2, MODE_I), Coordinate(4, -2, MODE_I),
            Coordinate(1, -1, MODE_I), Coordinate(2, -1, MODE_I), Coordinate(3, -1, MODE_I), Coordinate(4, -1, MODE_I),
            Coordinate(1,  0, MODE_I), Coordinate(2,  0, MODE_I), Coordinate(3,  0, MODE_I), Coordinate(4,  0, MODE_I),
            Coordinate(1,  1, MODE_I), Coordinate(2,  1, MODE_I), Coordinate(3,  1, MODE_I), Coordinate(4,  1, MODE_I)
        ]
        self.assertEqual(d.points_list('I'), expected)

    def test_larga_number_of_big_joins(self):
        start_time = time.time()
        d = DiscreteDisk.disk(radius=100)
        for i in range(10000):
            d.connect(100, i % 5, (i * 3) % 7)
        print(f"Operations: {discrete_disk.DiscreteDisk.get_operation_disk_counter()}")
        work_time = time.time() - start_time
        print(f"Time: {work_time} s")
        self.assertTrue(work_time < 5.0)

if __name__ == '__main__':
    unittest.main()
