import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from matplotlib.colors import ListedColormap
from typing import ClassVar, Iterator

# Binary representation for speed:
#   False -> forbidden / outer (previous MODE_O)
#   True  -> allowed  (previous MODE_I or MODE_B)
#
# Boundary mode is removed entirely (user request).

N = 2**10

DSQRT2 = 2 * np.sqrt(2)

_disk_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


@dataclass
class Options:
    crop: bool = False


opts = Options()


def R_CALC(mode: str = "1") -> None:
    """Calculate the ranges for discrete disks."""
    global RI, RE, RO, RIS, RES, ROS
    RI = np.empty(N, dtype="int64")
    RE = np.empty(N, dtype="int64")
    RO = np.empty(N, dtype="int64")
    RIS = np.empty(N, dtype="int64")
    RES = np.empty(N, dtype="int64")
    ROS = np.empty(N, dtype="int64")
    RI[0] = 0
    RE[0] = 0
    RO[0] = 1

    RIS[0] = 0
    RES[0] = 0
    ROS[0] = 2
    for i in range(1, N):
        if mode == "1":
            a = i**2
            b = i * DSQRT2
            c = 2
            RI[i] = np.floor(np.sqrt(a - b + c))
            RE[i] = i
            RO[i] = np.floor(np.sqrt(a + b + c))
            RIS[i] = np.floor(a - b + c)
            RES[i] = a
            ROS[i] = np.floor(a + b + c)
        else:
            RI[i] = i - 1
            RE[i] = i
            RO[i] = i + 1
            RIS[i] = RI[i] ** 2
            RES[i] = RE[i] ** 2
            ROS[i] = RO[i] ** 2


R_CALC(mode="1")

idx = lambda i, j: i * N - i * (i - 1) // 2 + (j - i)  # i<j


def D_CALC() -> None:
    global DS
    size = N * (N + 1) // 2
    DS = np.empty(size, dtype="int64")
    for i in range(N):
        for j in range(i, N):
            DS[idx(i, j)] = i**2 + j**2


D_CALC()


def symmetric_set(M: np.ndarray, i: int, j: int, radius: int, value: bool) -> None:
    """Set symmetric cells in matrix M (bool)."""
    M[radius + i, radius + j] = value
    M[radius + i, radius - j] = value
    M[radius - i, radius + j] = value
    M[radius - i, radius - j] = value
    M[radius + j, radius + i] = value
    M[radius + j, radius - i] = value
    M[radius - j, radius + i] = value
    M[radius - j, radius - i] = value


def _get_from_disk_cache(radius: int, connected: bool) -> np.ndarray:
    """
    Returns a boolean mask for a discrete disk of given radius.

    connected=True  -> disk interior+boundary are True, outside is False
    connected=False -> complement disk ("disconnect" mask): outside is True, interior+boundary are False
    """
    if radius not in _disk_cache:
        size = 2 * radius + 1

        # C: connected disk mask (True inside/boundary, False outside)
        C = np.full((size, size), False, dtype=np.bool_)

        # D: disconnected/complement mask (False inside/boundary, True outside)
        D = np.full((size, size), True, dtype=np.bool_)

        C[radius, radius] = True
        D[radius, radius] = False
        for ix in range(1, radius + 1):
            C[radius + ix, radius] = True
            C[radius - ix, radius] = True
            C[radius, radius + ix] = True
            C[radius, radius - ix] = True
            D[radius + ix, radius] = False
            D[radius - ix, radius] = False
            D[radius, radius + ix] = False
            D[radius, radius - ix] = False
            for iy in range(ix, radius + 1):
                if DS[idx(ix, iy)] <= RES[radius]:
                    symmetric_set(C, ix, iy, radius, True)
                    symmetric_set(D, ix, iy, radius, False)

        C.setflags(write=False)
        D.setflags(write=False)
        _disk_cache[radius] = (C, D)

    return _disk_cache[radius][0 if connected else 1]


@dataclass(slots=True)
class Coordinate:
    x: int
    y: int
    value: bool


@dataclass
class DiscreteDisk:
    """
    Boolean discrete disk / mask with a position (x,y).

    data: finite window of the mask (bool)
    rest: value outside the window (bool)
          - connected disk: rest=False
          - disconnected/complement disk: rest=True
    x,y: absolute coordinates of data[0,0] in the global grid
    """

    data: np.ndarray
    rest: bool
    x: int
    y: int
    _shared: bool = field(default=False, repr=False, compare=False)

    DISK_NONE: ClassVar["DiscreteDisk"]
    DISK_OUTER: ClassVar["DiscreteDisk"]
    DISK_INNER: ClassVar["DiscreteDisk"]

    operation_disk_counter: ClassVar[int] = 0

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
        rest = False if connected else True
        return cls(M, rest, x - radius, y - radius, True)

    def points_iter(self) -> Iterator[Coordinate]:
        """Iterate over True points (allowed cells) in this disk window."""
        ys, xs = np.nonzero(self.data)
        x0, y0 = self.x, self.y
        for iy, ix in zip(ys, xs):
            yield Coordinate(x0 + ix, y0 + iy, True)

    def points_list(self) -> list[Coordinate]:
        return list(self.points_iter())

    def shift(self, dx: int = 0, dy: int = 0) -> "DiscreteDisk":
        self.x += dx
        self.y += dy
        return self

    # --- Operations ---------------------------------------------------------
    # NOTE: With boolean masks we can unify operations:
    #   connect(r,x,y)    -> AND with connected disk (rest=False)
    #   disconnect(r,x,y) -> AND with complement disk (rest=True)
    #
    # Both are just logical_and on the overlap; the behavior outside overlap
    # depends only on b.rest.

    def connect(self, r: int, x: int, y: int) -> "DiscreteDisk":
        return self.operation_disk(DiscreteDisk.disk(r, x, y, connected=True))

    def connect_disk(self, b: "DiscreteDisk") -> "DiscreteDisk":
        return self.operation_disk(b)

    def disconnect(self, r: int, x: int = 0, y: int = 0) -> "DiscreteDisk":
        return self.operation_disk(DiscreteDisk.disk(r, x, y, connected=False))

    def disconnect_disk(self, b: "DiscreteDisk") -> "DiscreteDisk":
        return self.operation_disk(b)

    def operation_disk(self, b: "DiscreteDisk") -> "DiscreteDisk":
        """In-place logical AND with a shifted disk/mask b, keeping current window shape."""
        if self._shared:
            self.data = self.data.copy()
            self.data.setflags(write=True)
            self._shared = False

        ah, aw = self.data.shape
        ay, ax = self.y, self.x

        bh, bw = b.data.shape
        by, bx = b.y, b.x

        # Find overlap region in absolute coordinates
        oy1 = max(ay, by)
        ox1 = max(ax, bx)
        oy2 = min(ay + ah, by + bh)
        ox2 = min(ax + aw, bx + bw)

        if oy1 >= oy2 or ox1 >= ox2:
            # No overlap:
            # If b.rest is False, outside b is False => whole self becomes False after AND.
            if b.rest is False:
                self.data[:, :] = False
        else:
            # Apply AND on overlap (in-place, no allocation)
            asub = self.data[oy1 - ay : oy2 - ay, ox1 - ax : ox2 - ax]
            bsub = b.data[oy1 - by : oy2 - by, ox1 - bx : ox2 - bx]
            np.logical_and(asub, bsub, out=asub)

            # If outside b is False (connected disk), everything outside overlap
            # must also become False (rest-of-b behavior).
            if b.rest is False:
                self.data[: oy1 - ay, :] = False
                self.data[oy2 - ay :, :] = False
                self.data[:, : ox1 - ax] = False
                self.data[:, ox2 - ax :] = False

        DiscreteDisk.increment_operation_disk_counter()
        return self

    # --- Utility ------------------------------------------------------------

    def crop(self) -> "DiscreteDisk":
        """Crop the matrix by removing outer rows/columns equal to self.rest."""
        if not opts.crop:
            return self

        mask = self.data != self.rest

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return DISK_INNER if self.rest else DISK_OUTER

        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]

        self.data = self.data[y0 : y1 + 1, x0 : x1 + 1]
        self.x += x0
        self.y += y0
        return self

    def is_all_points_forbidden(self) -> bool:
        """Check if all points in the window are forbidden (False)."""
        return not np.any(self.data)

    def get_relative(self, x: int, y: int) -> bool:
        h, w = self.data.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return self.rest
        return bool(self.data[y, x])

    def get_absolute(self, x: int, y: int) -> bool:
        h, w = self.data.shape
        xr = x - self.x
        yr = y - self.y
        if xr < 0 or xr >= w or yr < 0 or yr >= h:
            return self.rest
        return bool(self.data[yr, xr])

    def show(self, symbol_map: np.ndarray = np.array(["◦", "█"])) -> str:
        """ASCII representation (False='◦', True='█')."""
        rows = ["".join(symbol_map[row.astype(np.int8)]) for row in self.data[::-1]]
        return "\n".join(rows)

    def display(self, ax=None) -> "plt.Axes":
        """Display the disk using matplotlib."""
        if ax is None:
            _, ax = plt.subplots()

        cmap = ListedColormap(["white", "red"])
        ax.imshow(self.data[::-1], interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        return ax


# Empty / sentinel disks
DISK_NONE = DiscreteDisk.disk(1, 0, 0)
DISK_OUTER = DiscreteDisk(np.full((0, 0), False, dtype=np.bool_), False, 0, 0, True)
DISK_INNER = DiscreteDisk(np.full((0, 0), True, dtype=np.bool_), True, 0, 0, True)


def create_area_by_join(a: DiscreteDisk, b: DiscreteDisk) -> DiscreteDisk:
    """Join area, increase window shape if needed (boolean version)."""
    DiscreteDisk.increment_operation_disk_counter()

    ah, aw = a.data.shape
    bh, bw = b.data.shape

    # Overlap region in absolute coords
    min_x = max(a.x, b.x)
    min_y = max(a.y, b.y)
    max_x = min(a.x + aw, b.x + bw)
    max_y = min(a.y + ah, b.y + bh)

    w = max_x - min_x
    h = max_y - min_y

    if w > 0 and h > 0:
        ax = min_x - a.x
        ay = min_y - a.y
        bx = min_x - b.x
        by = min_y - b.y
        M = np.logical_and(a.data[ay : ay + h, ax : ax + w], b.data[by : by + h, bx : bx + w])
    else:
        M = DISK_NONE

    # Cases depend only on rest values (infinite background)
    if (a.rest is False) and (b.rest is False):
        # Both connected-like (False outside)
        if M is DISK_NONE:
            return DISK_OUTER
        return DiscreteDisk(M, False, min_x, min_y, False).crop()

    if (a.rest is False) ^ (b.rest is False):
        # One connected (rest=False), other disconnected/complement (rest=True)
        if M is DISK_NONE:
            # No overlap: AND with True outside => connected disk unchanged
            return a if a.rest is False else b

        c = a if a.rest is False else b
        cx = min_x - c.x
        cy = min_y - c.y
        OM = c.data.copy()
        np.copyto(OM[cy : cy + h, cx : cx + w], M)
        return DiscreteDisk(OM, False, c.x, c.y, False).crop()

    # Both disconnected-like (True outside)
    min_x_oo = min(a.x, b.x)
    min_y_oo = min(a.y, b.y)
    max_x_oo = max(a.x + aw, b.x + bw)
    max_y_oo = max(a.y + ah, b.y + bh)

    w_oo = max_x_oo - min_x_oo
    h_oo = max_y_oo - min_y_oo

    MOO = np.full((h_oo, w_oo), True, dtype=np.bool_)

    ax_oo = a.x - min_x_oo
    ay_oo = a.y - min_y_oo
    np.copyto(MOO[ay_oo : ay_oo + ah, ax_oo : ax_oo + aw], a.data)

    bx_oo = b.x - min_x_oo
    by_oo = b.y - min_y_oo
    np.copyto(MOO[by_oo : by_oo + bh, bx_oo : bx_oo + bw], b.data)

    if M is not DISK_NONE:
        x_oo = min_x - min_x_oo
        y_oo = min_y - min_y_oo
        np.copyto(MOO[y_oo : y_oo + h, x_oo : x_oo + w], M)

    return DiscreteDisk(MOO, True, min_x_oo, min_y_oo, False).crop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discrete Disk Helper (binary masks).")

    parser.add_argument("-r", "--radius", type=int, default=4, help="Radius of the discrete disk (default: 4)")
    parser.add_argument("-p", "--print", action="store_true", help="Print area of the discrete disk")
    parser.add_argument("-d", "--display", action="store_true", help="Display the area of the discrete disk")

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
