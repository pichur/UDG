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

N=2**8

DSQRT2 = 2*np.sqrt(2)

_disk_cache: dict[int, (np.ndarray, np.ndarray)] = {}

def R_CALC():
    """Calculate the ranges for discrete disks."""
    global RI, RO, RIS, ROS
    RI  = np.empty(N, dtype='int64')
    RO  = np.empty(N, dtype='int64')
    RIS = np.empty(N, dtype='int64')
    ROS = np.empty(N, dtype='int64')
    RI [0] = 0
    RO [0] = 1
    RIS[0] = 0
    ROS[0] = 2
    for i in range(1,N):
        a = i**2
        b = i*DSQRT2
        c = 2
        RI [i] = np.floor(np.sqrt(a - b + c))
        RO [i] = np.floor(np.sqrt(a + b + c))
        RIS[i] = np.floor(a - b + c)
        ROS[i] = np.floor(a + b + c)
R_CALC()

idx = lambda i,j: i*N - i*(i+1)//2 + (j-i)  # i<j

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
    M[radius + i, radius + j] = symbol
    M[radius + i, radius - j] = symbol
    M[radius - i, radius + j] = symbol
    M[radius - i, radius - j] = symbol
    M[radius + j, radius + i] = symbol
    M[radius + j, radius - i] = symbol
    M[radius - j, radius + i] = symbol
    M[radius - j, radius - i] = symbol

def _get_from_disk_cache(radius: int, connected: bool) -> np.ndarray:
    if radius not in _disk_cache:
        r = radius + 1  # radius + margin=floor(sqrt(2))
        # + 1 for 0; C = Connected, D = Disconnected
        C = np.full((2 * r + 1, 2 * r + 1), MODE_I, dtype=np.uint8)
        D = np.full((2 * r + 1, 2 * r + 1), MODE_O, dtype=np.uint8)
        for ix in range(r + 1):
            for iy in range(ix, r + 1):
                if DS[idx(ix, iy)] >= ROS[radius]:
                    symmetric_set(C, ix, iy, r, MODE_O)
                    symmetric_set(D, ix, iy, r, MODE_I)
                elif DS[idx(ix, iy)] > RIS[radius]:
                    symmetric_set(C, ix, iy, r, MODE_B)
                    symmetric_set(D, ix, iy, r, MODE_B)

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
    
    @classmethod
    def disk(cls, radius: int = 4, x: int = 0, y: int = 0, connected: bool = True) -> "DiscreteDisk":
        r = radius + 1  # radius + margin=floor(sqrt(2))
        M = _get_from_disk_cache(radius, connected)
        return cls(M, MODE_O if connected else MODE_I, x - r, y - r, True)

    def points_iter(self, types: tuple[np.uint8, ...] = (MODE_I, MODE_B)):
        """Iterate over points of selected types.

        Parameters
        ----------
        types : tuple[np.uint8, ...], optional
            Cell types to iterate over. Defaults to ``(I, B)``.

        Yields
        ------
        tuple[int, int]
            ``(x, y)`` coordinates of matching cells, ordered row by row with
            ``y`` increasing first and ``x`` increasing second.
        """
        h, w = self.data.shape
        for iy in range(h):
            y = self.y + iy
            for ix in range(w):
                if self.data[iy, ix] in types:
                    x = self.x + ix
                    yield Coordinate(x, y, self.data[iy, ix])

    def points_IB_iter(self):
        h, w = self.data.shape
        for iy in range(h):
            y = self.y + iy
            for ix in range(w):
                if self.data[iy, ix] == MODE_I:
                    x = self.x + ix
                    yield Coordinate(x, y, self.data[iy, ix])
        for iy in range(h):
            y = self.y + iy
            for ix in range(w):
                if self.data[iy, ix] == MODE_B:
                    x = self.x + ix
                    yield Coordinate(x, y, self.data[iy, ix])

    def points_list(self, types: tuple[np.uint8, ...] = (MODE_I, MODE_B)) -> list[Coordinate]:
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

        h, w = self.data.shape

        # Calculate relative position of b in self's coordinates
        bx = b.x - self.x
        by = b.y - self.y

        # Find overlap region
        iay0 = max(0, by)
        iay1 = min(h, by + b.data.shape[0])
        iax0 = max(0, bx)
        iax1 = min(w, bx + b.data.shape[1])

        iby0 = max(0, -by)
        iby1 = min(h, -by + b.data.shape[0])
        ibx0 = max(0, -bx)
        ibx1 = min(w, -bx + b.data.shape[1])
        
        if iay0 >= iay1 or iax0 >= iax1:
            if operation is TBL_AND and b.rest == MODE_O:
                # If no overlap and rest of connected disk is outer, set all to outer
                self.data[:, :] = MODE_O
        else:
            # Apply the operation
            asub = self.data[iay0:iay1, iax0:iax1]          # widok, nie kopia
            bsub = b   .data[iby0:iby1, ibx0:ibx1]          # ten sam kształt co a
            np.copyto(asub, operation[asub, bsub])          # zapis in-place
            if operation is TBL_AND and b.rest == MODE_O:
                # If rest of connected disk is outer, rest of the disk set to outer too
                self.data[:iay0, :] = MODE_O
                self.data[iay1:, :] = MODE_O
                self.data[:, :iax0] = MODE_O
                self.data[:, iax1:] = MODE_O

        # Return self for method chaining
        return self
    
    def crop(self) -> "DiscreteDisk":
        """Crop the matrix by removing outer rows/columns with values equal to self.rest."""
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
        return self.data(y, x)
    
    def get_absolute(self, x: int, y: int) -> np.uint8:
        h, w = self.data.shape
        x_relative = x - self.x
        y_relative = y - self.y
        if x_relative < 0 or x_relative >= w or y_relative < 0 or y_relative >= h:
            return MODE_O
        return self.data(y, x)

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
            return DISK_OUTER
        else:
            o = a if a.rest == MODE_O else b
            ox = min_x - o.x
            oy = min_y - o.y
            OM = o.data.copy()
            np.copyto(OM[oy:oy+h, ox:ox+w], M)
            return DiscreteDisk(OM, MODE_O, o.x, o.y, False).crop()
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
