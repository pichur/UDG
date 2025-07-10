import time
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

def discrete_disk(radius: int) -> np.ndarray:
    r = radius + 3
    M = np.full((2*r, 2*r), I, dtype=np.uint8)
    for x in range(r):
        for y in range(x, r):
            if DS[idx(x,y)] > ROS[radius]:
                symmetric_set(M, x, y, r, O)
            elif DS[idx(x,y)] > RIS[radius]:
                symmetric_set(M, x, y, r, B)
    return M

def shift(M: np.ndarray, dx: int = 0, dy: int = 0, fill: np.uint8 = O) -> np.ndarray:
    """Zwraca tablicę przesuniętą o ``dx`` w poziomie i ``dy`` w pionie.

    W razie potrzeby tablica jest powiększana tak, aby cała zawartość mieściła
    się w nowym obszarze. Nowe komórki wypełniane są symbolem ``fill``.
    """
    h, w = M.shape
    top_pad = max(dy, 0)
    bottom_pad = max(-dy, 0)
    right_pad = max(dx, 0)
    left_pad = max(-dx, 0)
    result = np.full((h + top_pad + bottom_pad, w + left_pad + right_pad),
                     fill, dtype=M.dtype)
    result[top_pad:top_pad + h, right_pad:right_pad + w] = M
    return result


def meet(A: np.ndarray, B: np.ndarray, shift_b: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Zwraca wynik A ⊓ B w tym samym kodowaniu.

    Tablica ``B`` może zostać przesunięta o ``shift_b`` (``dx``, ``dy``) przed
    wykonaniem operacji. Funkcja obsługuje tablice o różnych rozmiarach i
    zwraca obszar obejmujący oba argumenty.
    """
    dx, dy = shift_b
    a_h, a_w = A.shape
    b_h, b_w = B.shape

    min_row = min(0, dy)
    min_col = min(0, dx)
    max_row = max(a_h, dy + b_h)
    max_col = max(a_w, dx + b_w)

    height = max_row - min_row
    width = max_col - min_col

    AA = np.full((height, width), O, dtype=np.uint8)
    BB = np.full((height, width), O, dtype=np.uint8)

    AA[-min_row:-min_row + a_h, -min_col:-min_col + a_w] = A
    BB[dy - min_row:dy - min_row + b_h, dx - min_col:dx - min_col + b_w] = B

    return AA & BB

def show(M: np.ndarray, symbol_map : np.ndarray = np.array(['◦', '▒', '?', '█'])) -> str:
    """█▓▒░∙◦ Zwraca widok n×n jako kwadrat zapełniony znakami: '█' dla I, '▒' dla B, '◦' dla O."""
    rows = [''.join(symbol_map[row]) for row in M[::-1]]
    return '\n'.join(rows)

def display(M: np.ndarray, ax=None) -> "plt.Axes":
    """Display the disk using :mod:`matplotlib`.

    Parameters
    ----------
    M : np.ndarray
        Matrix returned by :func:`discrete_disk`.
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
