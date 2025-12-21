import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from matplotlib.colors import ListedColormap
from typing import ClassVar

MODE_O = 0 # symbol 'O' (outer   ) 0
MODE_B = 1 # symbol 'B' (boundary) 1
MODE_I = 2 # symbol 'I' (interior) 2
MODE_U = 3 # symbol 'U' (unknown ) 3

MODES = ['O', 'B', 'I', '?']

 # kolejność kolumn:  b = O, B, I   (0,1,2)
TBL_AND  = np.array([[MODE_O,MODE_O,MODE_O],   # a = O    a & b  b & a
                     [MODE_O,MODE_B,MODE_B],   # a = B
                     [MODE_O,MODE_B,MODE_I]],  # a = I
                    dtype=np.uint8)
 # kolejność kolumn:  b = O, B, I   (0,1,2)
TBL_DIFF = np.array([[MODE_O,MODE_O,MODE_O],   # a = O    a \ b
                     [MODE_B,MODE_B,MODE_O],   # a = B
                     [MODE_I,MODE_B,MODE_O]],  # a = I
                    dtype=np.uint8)

N = 2**10
DSQRT2 = 2 * np.sqrt(2)

_disk_cache: dict[int, (np.ndarray, np.ndarray)] = {}

@dataclass
class Options:
    crop: bool = False
opts = Options()

def R_CALC(mode:str = '1'):
    """Calculate the ranges for discrete disks."""
    global RI, RE, RO, RIS, RES, ROS
    RI  = np.empty(N, dtype='int64')
    RE  = np.empty(N, dtype='int64')
    RO  = np.empty(N, dtype='int64')
    RIS = np.empty(N, dtype='int64')
    RES = np.empty(N, dtype='int64')
    ROS = np.empty(N, dtype='int64')
    RI [0] = 0
    RE [0] = 0
    RO [0] = 1

    RIS[0] = 0
    RES[0] = 0
    ROS[0] = 2
    for i in range(1,N):
        if mode == '1':
            a = i**2
            b = i*DSQRT2
            c = 2
            RI [i] = np.floor(np.sqrt(a - b + c))
            RE [i] = i
            RO [i] = np.floor(np.sqrt(a + b + c))
            RIS[i] = np.floor(a - b + c)
            RES[i] = a
            ROS[i] = np.floor(a + b + c)
        else:
            RI [i] = i - 1
            RE [i] = i
            RO [i] = i + 1
            RIS[i] = RI[i]**2
            RES[i] = RE[i]**2
            ROS[i] = RO[i]**2
R_CALC(mode='1')

idx = lambda i,j: i*N - i*(i-1)//2 + (j-i)  # i<j

def D_CALC():
    global DS
    size = N*(N+1)//2
    DS = np.empty(size, dtype='int64')
    for i in range(N):
        for j in range(i, N):
            DS[idx(i,j)] = i**2 + j**2
D_CALC()

def symmetric_set(M: np.ndarray, i: int, j: int, radius: int, symbol: np.uint8) -> None:
    """Ustawia symetryczne komórki w macierzy M."""
    M[radius + i    , radius + j    ] = symbol
    M[radius + i    , radius - j + 1] = symbol
    M[radius - i + 1, radius + j    ] = symbol
    M[radius - i + 1, radius - j + 1] = symbol
    M[radius + j    , radius + i    ] = symbol
    M[radius + j    , radius - i + 1] = symbol
    M[radius - j + 1, radius + i    ] = symbol
    M[radius - j + 1, radius - i + 1] = symbol

def _get_from_disk_cache(radius: int, connected: bool) -> np.ndarray:
    if radius not in _disk_cache:
        size = 2 * radius + 1
        # C = Connected, D = Disconnected
        C = np.full((size, size), MODE_O, dtype=np.uint8)
        D = np.full((size, size), MODE_I, dtype=np.uint8)
        for ix in range(1, radius + 1):
            for iy in range(ix, radius + 1):
                if DS[idx(ix, iy)] <= RES[radius]:
                    symmetric_set(C, ix, iy, radius, MODE_I)
                    symmetric_set(D, ix, iy, radius, MODE_O)
                else: # DS[idx(ix, iy)] > RES[radius]
                    if DS[idx(ix-1, iy-1)] < RES[radius]:
                        symmetric_set(C, ix, iy, radius, MODE_B)
                        symmetric_set(D, ix, iy, radius, MODE_B)
                    elif DS[idx(ix-1, iy-1)] == RES[radius]: # Not symmetric case caused by range (a,b> closed at Top Right
                        C[radius - ix + 1, radius - iy + 1] = MODE_B
                        C[radius - iy + 1, radius - ix + 1] = MODE_B
                        D[radius - ix + 1, radius - iy + 1] = MODE_B
                        D[radius - iy + 1, radius - ix + 1] = MODE_B
        # Not symmetric case caused by range (a,b> closed at Top Right, out of basic range
        C[0, radius] = MODE_B
        C[radius, 0] = MODE_B
        D[0, radius] = MODE_B
        D[radius, 0] = MODE_B

        if C[radius, radius] == MODE_I and not DiscreteDisk.allow_same_positions:
            C[radius, radius] = MODE_B

        C.setflags(write=False)
        D.setflags(write=False)
        _disk_cache[radius] = (C, D)
    
    return _disk_cache[radius][0 if connected else 1]


@dataclass(slots=True)
class Coordinate:
    x: int
    y: int
    mode: np.uint8

@dataclass
class DiscreteDisk:
    data: np.ndarray
    rest: np.uint8
    x: int
    y: int
    _shared: bool = field(default=False, repr=False, compare=False)

    DISK_NONE  : ClassVar["DiscreteDisk"]
    DISK_OUTER : ClassVar["DiscreteDisk"]
    DISK_INNER : ClassVar["DiscreteDisk"]

    operation_disk_counter: ClassVar[int] = 0
    
    allow_same_positions: ClassVar[bool] = True
    
    @classmethod
    def get_operation_disk_counter(cls) -> int:
        return cls.operation_disk_counter

    @classmethod
    def increment_operation_disk_counter(cls) -> int:
        cls.operation_disk_counter += 1
        return cls.operation_disk_counter

    @classmethod
    def reset_operation_disk_counter(cls) -> None:
        cls.operation_disk_counter = 0
    
    @classmethod
    def disk(cls, radius: int = 4, x: int = 0, y: int = 0, connected: bool = True) -> "DiscreteDisk":
        M = _get_from_disk_cache(radius, connected)
        return cls(M, MODE_O if connected else MODE_I, x - radius, y - radius, True)

    def points_iter(self, types: str = 'IB'):
        """Iterate over points of selected types."""
        if types == 'I':
            mask = (self.data == MODE_I) 
        elif types == 'B':
            mask = (self.data == MODE_B) 
        elif types == 'IB' or types == 'BI':
            mask = (self.data == MODE_I) | (self.data == MODE_B)
        else:
            raise ValueError('Not supported types: {types}')
        
        ys, xs = np.nonzero(mask)
        values = self.data[ys, xs]
        x0, y0 = self.x, self.y
        for iy, ix, val in zip(ys, xs, values):
            yield Coordinate(x0 + ix, y0 + iy, val)

    def points_IB_iter(self):
        self.points_iter('I')
        self.points_iter('B')

    def points_list(self, types: str = 'IB') -> list[Coordinate]:
        return list(self.points_iter(types))

    def points_IB_list(self) -> list[Coordinate]:
        return list(self.points_IB_iter())
    
    def shift(self, dx: int = 0, dy: int = 0) -> "DiscreteDisk":
        self.x += dx
        self.y += dy
        return self

    def connect(self, r: int, x: int, y: int) -> "DiscreteDisk":
        return self.operation(TBL_AND, r, x, y)

    def connect_disk(self, b: "DiscreteDisk") -> "DiscreteDisk":
        return self.operation_disk(TBL_AND, b)
    
    def disconnect(self, r: int, x: int = 0, y: int = 0) -> "DiscreteDisk":
        return self.operation(TBL_DIFF, r, x, y)

    def disconnect_disk(self, b: "DiscreteDisk") -> "DiscreteDisk":
        return self.operation_disk(TBL_DIFF, b)

    def operation(self, operation: np.ndarray, r: int, x: int, y: int) -> "DiscreteDisk":
        """Limit area by & with a shifted disk, keeping current area and shape."""
        return self.operation_disk(operation, DiscreteDisk.disk(r, x, y))

    def operation_disk(self, operation: np.ndarray, b: "DiscreteDisk") -> "DiscreteDisk":
        """Limit area by & with a shifted disk, keeping current area and shape."""
        if self._shared:
            self.data = self.data.copy()
            self.data.setflags(write=True)
            self._shared = False

        ah, aw = self.data.shape
        ay = self.y
        ax = self.x

        bh, bw = b.data.shape
        by = b.y
        bx = b.x

        # Find overlap region in relative coordinates
        oy1 = max(ay, by)
        ox1 = max(ax, bx)
        oy2 = min(ay + ah, by + bh)
        ox2 = min(ax + aw, bx + bw)

        if oy1 >= oy2 or ox1 >= ox2:
            if operation is TBL_AND and b.rest == MODE_O:
                # If no overlap and rest of connected disk is outer, set all to outer
                self.data[:, :] = MODE_O
        else:
            # Apply the operation
            asub = self.data[oy1-ay:oy2-ay, ox1-ax:ox2-ax]  # widok, nie kopia
            bsub = b   .data[oy1-by:oy2-by, ox1-bx:ox2-bx]  # ten sam kształt co a
            if operation is TBL_AND:
                np.minimum(asub, bsub, out=asub)   # in-place, bez alokacji
            else:
                np.copyto(asub, operation[asub, bsub])          # zapis in-place

            if operation is TBL_AND and b.rest == MODE_O:
                # If rest of connected disk is outer, rest of the disk set to outer too
                self.data[:oy1-ay, :] = MODE_O
                self.data[oy2-ay:, :] = MODE_O
                self.data[:, :ox1-ax] = MODE_O
                self.data[:, ox2-ax:] = MODE_O

        DiscreteDisk.increment_operation_disk_counter()

        # Return self for method chaining
        return self
    
    def crop(self) -> "DiscreteDisk":
        """Crop the matrix by removing outer rows/columns with values equal to self.rest."""
        if not opts.crop:
            return self
        
        mask = self.data != self.rest

        # Find bounds
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            # All values are rest, return minimal disk
            return DISK_INNER if self.rest == MODE_I else DISK_OUTER

        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]

        self.data = self.data[y0:y1+1, x0:x1+1]
        self.x += x0
        self.y += y0

        return self

    def is_all_points_O(self) -> bool:
        """Check if all points are set to the outer mode."""
        return np.all(self.data == MODE_O)
    
    def get_relative(self, x: int, y: int) -> np.uint8:
        h, w = self.data.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return MODE_O
        return self.data[y, x]
    
    def get_absolute(self, x: int, y: int) -> np.uint8:
        h, w = self.data.shape
        x_relative = x - self.x
        y_relative = y - self.y
        if x_relative < 0 or x_relative >= w or y_relative < 0 or y_relative >= h:
            return MODE_O
        return self.data[y, x]

    def show(self, symbol_map: np.ndarray = np.array(['◦', '▒', '█'])) -> str:
        """Return an ASCII representation of the matrix or :class:`DiscreteDisk`."""
        rows = [''.join(symbol_map[row]) for row in self.data[::-1]]
        return '\n'.join(rows)

    def display(self, ax=None) -> "plt.Axes":
        """Display the disk using :mod:`matplotlib`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes. If ``None``, a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the rendered image.
        """

        if ax is None:
            _, ax = plt.subplots()

        cmap = ListedColormap(['white', 'lightgray', 'red', 'black'])
        ax.imshow(self.data[::-1], interpolation='nearest', cmap=cmap, vmin=0, vmax=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        return ax

DISK_NONE  = DiscreteDisk.disk(1, 0, 0)
DISK_OUTER = DiscreteDisk(np.full((0, 0), MODE_O, dtype=np.uint8), MODE_O, 0, 0, True)
DISK_INNER = DiscreteDisk(np.full((0, 0), MODE_I, dtype=np.uint8), MODE_I, 0, 0, True)

def create_area_by_join(a: DiscreteDisk, b: DiscreteDisk) -> DiscreteDisk:
    """Join area, increase shape if need."""

    DiscreteDisk.increment_operation_disk_counter()

    ah, aw = a.data.shape
    bh, bw = b.data.shape
    
    # Basic operation, result is overlap region
    min_x = max(a.x     , b.x     )
    min_y = max(a.y     , b.y     )
    max_x = min(a.x + aw, b.x + bw)
    max_y = min(a.y + ah, b.y + bh)

    w = max_x - min_x
    h = max_y - min_y

    if w > 0 and h > 0:
        ax = min_x - a.x
        ay = min_y - a.y
        bx = min_x - b.x
        by = min_y - b.y
        M = TBL_AND[a.data[ay:ay+h, ax:ax+w], b.data[by:by+h, bx:bx+w]]
    else:
        M = DISK_NONE

    if a.rest == MODE_O and b.rest == MODE_O:
        # Both Outer
        if M is DISK_NONE:
            return DISK_OUTER
        else:
            return DiscreteDisk(M, MODE_O, min_x, min_y, False).crop()
    elif a.rest == MODE_O or b.rest == MODE_O:
        # One Outer other Inner
        if M is DISK_NONE:
            # Not overlapping, return the connected disk
            c = a if a.rest == MODE_O else b
            return c
        else:
            c = a if a.rest == MODE_O else b
            cx = min_x - c.x
            cy = min_y - c.y
            OM = c.data.copy()
            np.copyto(OM[cy:cy+h, cx:cx+w], M)
            return DiscreteDisk(OM, MODE_O, c.x, c.y, False).crop()
    else:
        # Both Inner
        min_x_oo = min(a.x     , b.x     )
        min_y_oo = min(a.y     , b.y     )
        max_x_oo = max(a.x + aw, b.x + bw)
        max_y_oo = max(a.y + ah, b.y + bh)

        w_oo = max_x_oo - min_x_oo
        h_oo = max_y_oo - min_y_oo
        
        MOO = np.full((h_oo, w_oo), MODE_I, dtype=np.uint8)

        ax_oo = a.x - min_x_oo
        ay_oo = a.y - min_y_oo
        np.copyto(MOO[ay_oo:ay_oo+ah, ax_oo:ax_oo+aw], a.data)

        bx_oo = b.x - min_x_oo
        by_oo = b.y - min_y_oo
        np.copyto(MOO[by_oo:by_oo+bh, bx_oo:bx_oo+bw], b.data)

        if M is not DISK_NONE:
            x_oo = min_x - min_x_oo
            y_oo = min_y - min_y_oo
            np.copyto(MOO[y_oo:y_oo+h, x_oo:x_oo+w], M)

        return DiscreteDisk(MOO, MODE_I, min_x_oo, min_y_oo, False).crop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discrete Disk Helper.")

    parser.add_argument(
        "-r", "--radius", type=int, default=4,
        help="Radius of the discrete disk (default: 4)"
    )
    parser.add_argument(
        "-p", "--print", action="store_true",
        help="Print area of the discrete disk")
    parser.add_argument(
        "-d", "--display", action="store_true",
        help="Display the area of the discrete disk")
    
    args = parser.parse_args()

    s = time.perf_counter()
    R_CALC()
    s = time.perf_counter() - s
    print(f"Time R_CALC: {s:.6f} seconds")

    s = time.perf_counter()
    D_CALC()
    s = time.perf_counter() - s
    print(f"Time D_CALC: {s:.6f} seconds")

    s = time.perf_counter()
    a = DiscreteDisk.disk(args.radius)
    s = time.perf_counter() - s
    print(f"Time discrete_area({args.radius}): {s:.6f} seconds")

    if args.print:
        print(a.show())

    if args.display:
        a.display()
        plt.show()
