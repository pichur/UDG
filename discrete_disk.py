import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.colors import ListedColormap

O = 0 # symbol 'O' (outer   ) 0
B = 1 # symbol 'B' (boundary) 1
I = 2 # symbol 'I' (interior) 2

 # kolejność kolumn:  b = O, B, I   (0,1,2)
AND_TBL  = np.array([[O,O,O],   # a = O    a & b  b & a
                     [O,B,B],   # a = B
                     [O,B,I]],  # a = I
                    dtype=np.uint8)
 # kolejność kolumn:  b = O, B, I   (0,1,2)
DIFF_TBL = np.array([[O,O,O],   # a = O    a \ b
                     [B,B,O],   # a = B
                     [I,B,O]],  # a = I
                    dtype=np.uint8)

N=2**8

DSQRT2 = 2*np.sqrt(2)

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

@dataclass
class DiscreteDisk:
    data: np.ndarray
    x: int
    y: int
    
    @classmethod
    def disk(cls, radius: int = 4, x: int = 0, y: int = 0) -> "DiscreteDisk":
        r = radius + 1  # radius + margin=floor(sqrt(2))
        M = np.full((2 * r + 1, 2 * r + 1), I, dtype=np.uint8) # + 1 for 0
        for ix in range(r+1):
            for iy in range(ix, r+1):
                if DS[idx(ix, iy)] >= ROS[radius]:
                    symmetric_set(M, ix, iy, r, O)
                elif DS[idx(ix, iy)] > RIS[radius]:
                    symmetric_set(M, ix, iy, r, B)
        return cls(M, x - r, y - r)

    def iter_points(self, types: tuple[np.uint8, ...] = (I, B)):
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
                    yield (x, y)

    def shift(self, dx: int = 0, dy: int = 0) -> "DiscreteDisk":
        self.x += dx
        self.y += dy
        return self

    def connect(self, r: int, x: int, y: int) -> "DiscreteDisk":
        return self.operation(AND_TBL, r, x, y)

    def connect_disk(self, b: "DiscreteDisk") -> "DiscreteDisk":
        return self.operation_disk(AND_TBL, b)
    
    def disconnect(self, r: int, x: int = 0, y: int = 0) -> "DiscreteDisk":
        return self.operation(DIFF_TBL, r, x, y)

    def disconnect_disk(self, b: "DiscreteDisk") -> "DiscreteDisk":
        return self.operation_disk(DIFF_TBL, b)

    def operation(self, operation: np.ndarray, r: int, x: int, y: int) -> "DiscreteDisk":
        """Limit area by & with a shifted disk, keeping current area and shape."""
        return self.operation_disk(operation, DiscreteDisk.disk(r, x, y))

    def operation_disk(self, operation: np.ndarray, b: "DiscreteDisk") -> "DiscreteDisk":
        """Limit area by & with a shifted disk, keeping current area and shape."""
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
            if operation is AND_TBL:
                # If no overlap, set all to outer
                self.data[:, :] = O
        else:
            # Apply the operation
            asub = self.data[iay0:iay1, iax0:iax1]          # widok, nie kopia
            bsub = b   .data[iby0:iby1, ibx0:ibx1]          # ten sam kształt co a
            np.copyto(asub, operation[asub, bsub])          # zapis in-place
            if operation is AND_TBL:
                # Rest of the disk set to outer
                self.data[:iay0, :] = O
                self.data[iay1:, :] = O
                self.data[:, :iax0] = O
                self.data[:, iax1:] = O

        # Return self for method chaining
        return self

    def is_all_points_O(self) -> bool:
        """Check if all points are set to the outer type."""
        return np.all(self.data == O)
    
    def get_relative(self, x: int, y: int) -> np.uint8:
        h, w = self.data.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return O
        return self.data(y, x)
    
    def get_absolute(self, x: int, y: int) -> np.uint8:
        h, w = self.data.shape
        x_relative = x - self.x
        y_relative = y - self.y
        if x_relative < 0 or x_relative >= w or y_relative < 0 or y_relative >= h:
            return O
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
