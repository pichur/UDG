import numpy as np

N=2**8

DSQRT2 = 2*np.sqrt(2)

RIS = np.empty(N, dtype='int64')
ROS = np.empty(N, dtype='int64')
RIS[0] = 0
ROS[0] = 2
for i in range(1,N):
    c = i**2 + 2
    d = np.floor(i*DSQRT2)
    RIS[i] = c - d
    ROS[i] = c + d

size = N*(N+1)//2
DS = np.empty(size, dtype='int64')
idx = lambda i,j: i*N - i*(i+1)//2 + (j-i)  # i<j
for i in range(N):
    for j in range(i, N):
        DS[idx(i,j)] = i**2 + j**2

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
    for i in range(r):
        for j in range(i, r):
            if DS[idx(i,j)] > ROS[radius]:
                symmetric_set(M, i, j, r, O)
            elif DS[idx(i,j)] > RIS[radius]:
                symmetric_set(M, i, j, r, B)
    return M

I = 0b11 # symbol 'I' (interior) 3
B = 0b01 # symbol 'B' (boundary) 1
O = 0b00 # symbol 'O' (outer   ) 0

# Mapowanie kodów na litery dla podglądu
DEC = np.array(['O', 'B', 'x', 'I'])

def new_table(n: int, fill=O) -> np.ndarray:
    """Zwraca tablicę n×n wypełnioną symbolem fill (I/B/O)."""
    return np.full((n, n), fill, dtype=np.uint8)

def meet(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Zwraca wynik A ⊓ B w tym samym kodowaniu."""
    return A & B      # działa wektorowo i natychmiast

def show(M: np.ndarray, symbol_map : np.ndarray = np.array(['◦', '▒', '?', '█'])) -> str:
    """█▓▒░∙◦ Zwraca widok n×n jako kwadrat zapełniony znakami: '█' dla I, '▒' dla B, '◦' dla O."""
    rows = [''.join(symbol_map[row]) for row in M]
    return '\n'.join(rows)

if __name__ == "__main__":
    n = 16
    A = discrete_disk(n)  # A = dysk o promieniu n

    aaaa = show(A)
    print("A:")
    print(aaaa)
    # print("B:\n", show(B))
    # print("C = A ⊓ B:\n", show(C))


# if __name__ == "__main__":
#     radius = 4
#     points = discrete_disk(radius)
#     print(f"Points in discrete disk of radius {radius}:")
#     for point in points:
#         print(point)