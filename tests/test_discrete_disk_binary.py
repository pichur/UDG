import unittest
import os, sys
import textwrap
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from discrete_disk_binary import DiscreteDisk, Coordinate, DISK_INNER, DISK_OUTER, create_area_by_join
import discrete_disk_binary

TEST_SHOW = np.array(['-', '+'])

class TestDiscreteDiskBinary(unittest.TestCase):

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
        d = DiscreteDisk.disk(radius=1, connected=True)
        self.assertDDDEq(d, """
        -+-
        +++
        -+-
        """)

    def test_content_c2(self):
        d = DiscreteDisk.disk(radius=2, connected=True)
        self.assertDDDEq(d, """
        --+--
        -+++-
        +++++
        -+++-
        --+--
        """)

    def test_content_c3(self):
        d = DiscreteDisk.disk(radius=3, connected=True)
        self.assertDDDEq(d, """
        ---+---
        -+++++-
        -+++++-
        +++++++
        -+++++-
        -+++++-
        ---+---
        """)

    def test_content_c4(self):
        d = DiscreteDisk.disk(radius=4, connected=True)
        self.assertDDDEq(d, """
        ----+----
        --+++++--
        -+++++++-
        -+++++++-
        +++++++++
        -+++++++-
        -+++++++-
        --+++++--
        ----+----
        """)

    def test_content_d4(self):
        d = DiscreteDisk.disk(radius=4, connected=False)
        self.assertDDDEq(d, """
        ++++-++++
        ++-----++
        +-------+
        +-------+
        ---------
        +-------+
        +-------+
        ++-----++
        ++++-++++
        """)

    def test_connect_a(self):
        d = DiscreteDisk.disk(4)
        d.connect(4, 2, 0)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertDDDEq(d, """
        ---------
        ----+++--
        ---+++++-
        ---+++++-
        --+++++++
        ---+++++-
        ---+++++-
        ----+++--
        ---------
        """)

    def test_connect_b(self):
        d = DiscreteDisk.disk(4)
        d.connect(4, -2, -1)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertDDDEq(d, """
        ---------
        --+------
        -++++----
        -+++++---
        ++++++---
        -++++++--
        -+++++---
        --++++---
        ----+----
        """)

    def test_connect_c(self):
        d = DiscreteDisk.disk(4)
        d.connect(4, 3, 3)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertDDDEq(d, """
        ----+----
        ---++++--
        ----++++-
        ----++++-
        -----++++
        -------+-
        ---------
        ---------
        ---------
        """)

    def test_connect_d(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, 8, 8)
        self.assertEqual(d.is_all_points_forbidden(), True)

    def test_connect_out_of_range_a(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, 10, -2)
        self.assertEqual(d.is_all_points_forbidden(), True)

    def test_connect_out_of_range_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, -9, 9)
        self.assertEqual(d.is_all_points_forbidden(), True)

    def test_disconnect_a(self):
        d = DiscreteDisk.disk(4)
        d.disconnect(4, 2, 0)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertDDDEq(d, """
        ----+----
        --++-----
        -++------
        -++------
        ++-------
        -++------
        -++------
        --++-----
        ----+----
        """)

    def test_disconnect_b(self):
        d = DiscreteDisk.disk(4)
        d.disconnect(4, -2, -1)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertDDDEq(d, """
        ----+----
        ---++++--
        -----+++-
        ------++-
        ------+++
        -------+-
        ------++-
        ------+--
        ---------
        """)

    def test_disconnect_out_of_range_a(self):
        d = DiscreteDisk.disk(4)
        d.disconnect(3, 10, -2)
        self.assertDDDEq(d, """
        ----+----
        --+++++--
        -+++++++-
        -+++++++-
        +++++++++
        -+++++++-
        -+++++++-
        --+++++--
        ----+----
        """)

    def test_disconnect_out_of_range_b(self):
        d = DiscreteDisk.disk(3)
        r = 4
        d.connect(3, -9, 9)
        self.assertEqual(d.is_all_points_forbidden(), True)
    
    def test_create_area_by_join_CC(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -7, y =  1, connected = True),
            b = DiscreteDisk.disk(radius = 4, x = -6, y = -1, connected = True))
        self.assertEqual(d.x, -10) 
        self.assertEqual(d.y, - 3)
        self.assertEqual(d.data.shape, (7, 8))
        self.assertEqual(d.rest, False)
        self.assertDDDEq(d, """
        ----+---
        --+++++-
        -+++++++
        -++++++-
        +++++++-
        -+++++--
        ---+----
        """)

    def test_create_area_by_join_CC_out_of_range(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 3, x = -8, y =  9, connected = True),
            b = DiscreteDisk.disk(radius = 3, x =  7, y = -5, connected = True))
        self.assertIs(d, DISK_OUTER)

    def test_create_area_by_join_CD(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -7, y =  1, connected = True),
            b = DiscreteDisk.disk(radius = 4, x = -6, y = -1, connected = False))
        self.assertEqual(d.x, -7 - 4) 
        self.assertEqual(d.y,  1 - 4)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertEqual(d.rest, False)
        self.assertDDDEq(d, """
        ----+----
        --+++++--
        -++++-++-
        -++------
        ++-------
        -+-------
        ---------
        ---------
        ---------
        """)

    def test_create_area_by_join_CD_crop(self):
        discrete_disk_binary.opts.crop = True
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -7, y =  1, connected = True),
            b = DiscreteDisk.disk(radius = 4, x = -6, y = -1, connected = False))
        discrete_disk_binary.opts.crop = False
        self.assertEqual(d.x, -7 - 4) 
        self.assertEqual(d.y,  1 - 4 + 3)
        self.assertEqual(d.data.shape, (6, 8))
        self.assertEqual(d.rest, False)
        self.assertDDDEq(d, """
        ----+---
        --+++++-
        -++++-++
        -++-----
        ++------
        -+------
        """)

    def test_create_area_by_join_DC(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -7, y =  1, connected = False),
            b = DiscreteDisk.disk(radius = 4, x = -6, y = -1, connected = True))
        self.assertEqual(d.x, -6 - 4)
        self.assertEqual(d.y, -1 - 4)
        self.assertEqual(d.data.shape, (9, 9))
        self.assertEqual(d.rest, False)
        self.assertDDDEq(d, """
        ---------
        ---------
        ---------
        -------+-
        -------++
        ------++-
        -++-++++-
        --+++++--
        ----+----
        """)

    def test_create_area_by_join_DC_crop(self):
        discrete_disk_binary.opts.crop = True
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -7, y =  1, connected = False),
            b = DiscreteDisk.disk(radius = 4, x = -7, y = -3, connected = True))
        discrete_disk_binary.opts.crop = False
        self.assertEqual(d.x, -7 - 4)
        self.assertEqual(d.y, -3 - 4)
        self.assertEqual(d.data.shape, (6, 9))
        self.assertEqual(d.rest, False)
        self.assertDDDEq(d, """
        -+-----+-
        ++++-++++
        -+++++++-
        -+++++++-
        --+++++--
        ----+----
        """)

    def test_create_area_by_join_DD(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -7, y =  1, connected = False),
            b = DiscreteDisk.disk(radius = 4, x = -6, y = -1, connected = False))
        self.assertEqual(d.x, -7 - 4)
        self.assertEqual(d.y, -1 - 4)
        self.assertEqual(d.data.shape, (11, 10))
        self.assertEqual(d.rest, True)
        self.assertDDDEq(d, """
        ++++-+++++
        ++-----+++
        +-------++
        +-------++
        ---------+
        +--------+
        +---------
        ++-------+
        ++-------+
        +++-----++
        +++++-++++
        """)

    def test_create_area_by_join_DD_out_of_range(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = -8, y =  9, connected = False),
            b = DiscreteDisk.disk(radius = 4, x =  3, y = -1, connected = False))
        self.assertEqual(d.x, -12)
        self.assertEqual(d.y, - 5)
        self.assertEqual(d.data.shape, (19, 20))
        self.assertEqual(d.rest, True)
        self.assertDDDEq(d, """
        ++++-+++++++++++++++
        ++-----+++++++++++++
        +-------++++++++++++
        +-------++++++++++++
        ---------+++++++++++
        +-------++++++++++++
        +-------++++++++++++
        ++-----+++++++++++++
        ++++-+++++++++++++++
        ++++++++++++++++++++
        +++++++++++++++-++++
        +++++++++++++-----++
        ++++++++++++-------+
        ++++++++++++-------+
        +++++++++++---------
        ++++++++++++-------+
        ++++++++++++-------+
        +++++++++++++-----++
        +++++++++++++++-++++
        """)

    def test_create_area_by_join_D3C6_0(self):
        d = create_area_by_join(
            a = DiscreteDisk.disk(radius = 4, x = 0, y = 0, connected = False),
            b = DiscreteDisk.disk(radius = 7, x = 0, y = 0, connected = True ))
        self.assertEqual(d.x, -7)
        self.assertEqual(d.y, -7)
        self.assertEqual(d.data.shape, (15, 15))
        self.assertEqual(d.rest, False)

        self.assertDDDEq(d, """
        -------+-------
        ----+++++++----
        ---+++++++++---
        --+++++-+++++--
        -++++-----++++-
        -+++-------+++-
        -+++-------+++-
        +++---------+++
        -+++-------+++-
        -+++-------+++-
        -++++-----++++-
        --+++++-+++++--
        ---+++++++++---
        ----+++++++----
        -------+-------
        """)
    
    def test_iter_points(self):
        d = DiscreteDisk.disk(4)
        p = d.points_list()

        expected = [
                                                                                                                    Coordinate(0, -4, True),
                                                                Coordinate(-2, -3, True), Coordinate(-1, -3, True), Coordinate(0, -3, True), Coordinate(1, -3, True), Coordinate(2, -3, True),
                                      Coordinate(-3, -2, True), Coordinate(-2, -2, True), Coordinate(-1, -2, True), Coordinate(0, -2, True), Coordinate(1, -2, True), Coordinate(2, -2, True), Coordinate(3, -2, True),
                                      Coordinate(-3, -1, True), Coordinate(-2, -1, True), Coordinate(-1, -1, True), Coordinate(0, -1, True), Coordinate(1, -1, True), Coordinate(2, -1, True), Coordinate(3, -1, True),
            Coordinate(-4,  0, True), Coordinate(-3,  0, True), Coordinate(-2,  0, True), Coordinate(-1,  0, True), Coordinate(0,  0, True), Coordinate(1,  0, True), Coordinate(2,  0, True), Coordinate(3,  0, True), Coordinate(4,  0, True),
                                      Coordinate(-3,  1, True), Coordinate(-2,  1, True), Coordinate(-1,  1, True), Coordinate(0,  1, True), Coordinate(1,  1, True), Coordinate(2,  1, True), Coordinate(3,  1, True),
                                      Coordinate(-3,  2, True), Coordinate(-2,  2, True), Coordinate(-1,  2, True), Coordinate(0,  2, True), Coordinate(1,  2, True), Coordinate(2,  2, True), Coordinate(3,  2, True),
                                                                Coordinate(-2,  3, True), Coordinate(-1,  3, True), Coordinate(0,  3, True), Coordinate(1,  3, True), Coordinate(2,  3, True),
                                                                                                                    Coordinate(0,  4, True)
        ]
        self.assertEqual(d.points_list(), expected)

if __name__ == '__main__':
    unittest.main()
