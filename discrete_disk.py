import time
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

N=2**8

DSQRT2 = 2*np.sqrt(2)

def R_CALC():
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
        b = i*DSQRT2
        c = i**2 + 2
        d = np.floor(b)
        RI [i] = np.floor(np.sqrt(c - b))
        RO [i] = np.floor(np.sqrt(c + b))
        RIS[i] = c - d
        ROS[i] = c + d
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

I = 0b11 # symbol 'I' (interior) 3
B = 0b01 # symbol 'B' (boundary) 1
O = 0b00 # symbol 'O' (outer   ) 0


@dataclass
class DiscreteDisk:
    data: np.ndarray
    pos: tuple[int, int]

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
        for j in range(h):
            y = self.pos[1] + j
            for i in range(w):
                if self.data[j, i] in types:
                    x = self.pos[0] + i
                    yield (x, y)

def symmetric_set(M: np.ndarray, i: int, j: int, radius: int, symbol: np.uint8) -> None:
    """Ustawia symetryczne komórki w macierzy M."""
    M[radius + i    , radius + j    ] = symbol
    M[radius + i    , radius - j - 1] = symbol
    M[radius - i - 1, radius + j    ] = symbol
    M[radius - i - 1, radius - j - 1] = symbol
    M[radius + j    , radius + i    ] = symbol
    M[radius + j    , radius - i - 1] = symbol
    M[radius - j - 1, radius + i    ] = symbol
    M[radius - j - 1, radius - i - 1] = symbol

def discrete_disk(radius: int) -> DiscreteDisk:
    r = radius + 3
    M = np.full((2 * r, 2 * r), I, dtype=np.uint8)
    for x in range(r):
        for y in range(x, r):
            if DS[idx(x,y)] > ROS[radius]:
                symmetric_set(M, x, y, r, O)
            elif DS[idx(x,y)] > RIS[radius]:
                symmetric_set(M, x, y, r, B)
    return DiscreteDisk(M, (-r, -r))

def shift(D: DiscreteDisk, dx: int = 0, dy: int = 0) -> DiscreteDisk:
    """Return a shifted copy of ``D``.

    Only the ``pos`` attribute is changed. The underlying matrix is shared
    between the returned object and ``D``.
    """
    return DiscreteDisk(D.data, (D.pos[0] + dx, D.pos[1] + dy))


def meet(A: DiscreteDisk, B: DiscreteDisk, shift_b: tuple[int, int] = (0, 0)) -> DiscreteDisk:
    """Return ``A`` ⊓ ``B`` taking positions into account.

    Parameters
    ----------
    A, B : DiscreteDisk
        Input disks. ``B`` can be additionally shifted by ``shift_b``.
    shift_b : tuple[int, int], optional
        Additional shift applied to ``B`` before intersection.
    """
    dx, dy = shift_b
    b_pos = (B.pos[0] + dx, B.pos[1] + dy)
    a_h, a_w = A.data.shape
    b_h, b_w = B.data.shape

    min_row = min(A.pos[1], b_pos[1])
    min_col = min(A.pos[0], b_pos[0])
    max_row = max(A.pos[1] + a_h, b_pos[1] + b_h)
    max_col = max(A.pos[0] + a_w, b_pos[0] + b_w)

    height = max_row - min_row
    width = max_col - min_col

    AA = np.full((height, width), O, dtype=np.uint8)
    BB = np.full((height, width), O, dtype=np.uint8)

    A_off_r = A.pos[1] - min_row
    A_off_c = A.pos[0] - min_col
    AA[A_off_r:A_off_r + a_h, A_off_c:A_off_c + a_w] = A.data

    B_off_r = b_pos[1] - min_row
    B_off_c = b_pos[0] - min_col
    BB[B_off_r:B_off_r + b_h, B_off_c:B_off_c + b_w] = B.data

    return DiscreteDisk(AA & BB, (min_col, min_row))

def show(M, symbol_map: np.ndarray = np.array(['◦', '▒', '?', '█'])) -> str:
    """Return an ASCII representation of the matrix or :class:`DiscreteDisk`."""
    if isinstance(M, DiscreteDisk):
        M = M.data
    rows = [''.join(symbol_map[row]) for row in M[::-1]]
    return '\n'.join(rows)

def display(M, ax=None) -> "plt.Axes":
    """Display the disk using :mod:`matplotlib`.

    Parameters
    ----------
    M : np.ndarray or DiscreteDisk
        Matrix or :class:`DiscreteDisk` instance returned by
        :func:`discrete_disk`.
    ax : matplotlib.axes.Axes, optional
        Target axes. If ``None``, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the rendered image.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if ax is None:
        _, ax = plt.subplots()

    if isinstance(M, DiscreteDisk):
        M = M.data

    cmap = ListedColormap(['white', 'lightgray', 'red', 'black'])
    ax.imshow(M[::-1], interpolation='nearest', cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    return ax

if __name__ == "__main__":
    s = time.perf_counter()
    R_CALC()
    s = time.perf_counter() - s
    print(f"Time R_CALC: {s:.6f} seconds")

    s = time.perf_counter()
    D_CALC()
    s = time.perf_counter() - s
    print(f"Time D_CALC: {s:.6f} seconds")

    n = 32

    s = time.perf_counter()
    a = discrete_disk(n)  # A = dysk o promieniu n
    s = time.perf_counter() - s
    print(f"Time discrete_disk({n}): {s:.6f} seconds")

    s = time.perf_counter()
    b = meet(a, a, (16, 0))
    s = time.perf_counter() - s

    print = False
    display = False

    if print:
        print("A:")
        print(show(a))

        print("A shift(2,3):")
        print(show(shift(a, 2, 3)))

        print("A shift(-3,-4):")
        print(show(shift(a, -3, -4)))

        print("A shift(-3,-4) & A shift(2,3):")
        print(show(meet(shift(a, -3, -4), shift(a, 2, 3))))

        print("A & A->(16,0):")
        print(f"Time meet(a, a, (16, 0)): {s:.6f} seconds")
        print(show(b))

    if display:
        display(b)
        plt.show()
