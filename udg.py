"""Simple Unit Disk Graph recognition algorithm."""

from math import sqrt, sin, cos, pi
from collections import deque
import networkx as nx
import argparse
import time
import Graph6Converter
import numpy as np
import matplotlib.pyplot as plt
import discrete_disk
from discrete_disk import create_area_by_join, DiscreteDisk, Coordinate, MODES, MODE_U, MODE_I, MODE_B, MODE_O, DISK_NONE
from dataclasses import dataclass


TRIGRAPH = "TRIGRAPH"
YES = "YES"
NO = "NO"

@dataclass(slots=True)
class IterationInfo:
    point_iter: int
    point_size: int

class Graph:
    """Simple adjacency list graph that can be built from an integer number
    of vertices or from a :class:`networkx.Graph` instance."""
    verbose: bool
    limit_points:bool = True

    unit: int
    n: int
    adj: list[list[int]]

    order_mode: str = 'DD'
    order: list[int]
    coordinates: list[Coordinate]

    previous_area: list[list[DiscreteDisk]]

    iteretions: list[IterationInfo]

    def __init__(self, n_or_g):
        if isinstance(n_or_g, int):
            self.n = n_or_g
            self.adj = [[] for _ in range(self.n)]
        elif isinstance(n_or_g, nx.Graph):
            mapping = {node: idx for idx, node in enumerate(sorted(n_or_g.nodes()))}
            self.n = len(mapping)
            self.adj = [[] for _ in range(self.n)]
            for u, v in n_or_g.edges():
                self.add_edge(mapping[u], mapping[v])
        else:
            raise TypeError("Graph() expects an int or a networkx.Graph")
        
        self.verbose = False

        # store vertex coordinates and additional parameters
        self.coordinates = [
            Coordinate(np.int64(0), np.int64(0), MODE_U)
            for _ in range(self.n)
        ]

        self.iteretions = [
            IterationInfo(0, 0)
            for _ in range(self.n)
        ]

        # set to real values during processing
        # indexing by permuted work order
        self.previous_area = [[DISK_NONE for _ in range(self.n)] for _ in range(self.n)]

        self.order = range(self.n)

        self.set_unit(1)
        

        # By theory the eps_min is 1/(2**(2**O(n)))
        ##self.eps_min = 1/(2**(2**self.n))
        # 1 -> 1/(2**(2**1)) = 1/4
        # 2 -> 1/(2**(2**2)) = 1/16
        # 3 -> 1/(2**(2**3)) = 1/256
        # 4 -> 1/(2**(2**4)) = 1/65536
        # 5 -> 1/(2**(2**5)) = 1/4294967296
        # self.eps_min = 1e-12
        self.eps_min = 1

    def add_edge(self, u: int, v: int):
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)

    def neighbors(self, v: int):
        return self.adj[v]

    def is_edge(self, u: int, v: int):
        return v in self.adj[u]
    
    def set_verbose(self, verbose):
        """Set verbosity for debugging purposes."""
        self.verbose = verbose
        return self
    
    def set_coordinate(self, v: int, x: int, y: int, mode: np.uint8 = MODE_U):
        """Set coordinates for vertex ``v``."""
        v = self.coordinates[v]
        v.x, v.y, v.mode = x, y, mode
        return self
    
    def clear_previous_area(self, order_index: int):
        fill = [DISK_NONE] * (self.n - order_index)
        for row in self.previous_area:
            row[order_index:] = fill

    def set_iteration_len(self, v: int, len: int):
        c = self.iteretions[v].point_size = len
        return self

    def set_iteration_index(self, v: int, index: int):
        c = self.iteretions[v].point_iter = index
        return self

    def set_unit(self, unit: int):
        """Set the unit disk radius for the graph."""
        self.unit = unit
        self.unit_sq = self.unit * self.unit
        return self
    
    def vertex_distance_squared(self, u: int, v: int) -> int:
        """Return squared Euclidean distance between vertices ``u`` and ``v``."""
        dx = self.coordinates[u].x - self.coordinates[v].x
        dy = self.coordinates[u].y - self.coordinates[v].y
        return dx * dx + dy * dy

    def vertex_distance(self, u: int, v: int) -> float:
        """Return Euclidean distance between vertices ``u`` and ``v``."""
        return sqrt(self.vertex_distance_squared(u, v))

    def print_result(self, print_vertex: bool, print_edges: bool) -> None:
        time = self.stop_time - self.start_time
        print(f"Time consumed: {time} s")
        if (print_vertex):
            for i in range(self.n):
                print(f"  V ({i}): ({self.coordinates[i].x:7d}, {self.coordinates[i].y:7d})")
        fail = False
        for i in range(self.n):
            for j in range(i + 1, self.n):
                e = self.is_edge(i, j)
                edisp = "E" if e else " "
                d_sq = self.vertex_distance_squared(i, j)
                dlter = d_sq <= self.unit_sq
                comp = "<=" if dlter else "> "
                d = sqrt(d_sq)
                if (print_edges):
                    print(f"  {edisp} dist({i},{j}) {d:10.3f} {comp} {self.unit:7d} ")
                if not fail:
                    fail = (e and not dlter) or (not e and dlter)
        if fail:
            print("FAIL: some edges are not in the unit disk range or some non-edges are in the range.")

    def print_vertex_distances(self):
        # Calculate max range
        print()
        print(f'result_I = {self.result_I:7d}')
        print(f'result_B = {self.result_B:7d}')

        print()
        max_distance = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                for d in range(discrete_disk.RO[self.unit] * self.n):
                    if self.vertex_distances[i][j][d] in (MODE_I, MODE_B):
                        max_distance = max(max_distance, d)
        header = f"  {self.n:3d} x {self.unit:3d} >"
        for d in range(1, max_distance):
            if d % self.unit == 0:
                c = '|'
            elif d % 10 == 0:
                c = '*'
            elif d % 5 == 0:
                c = '.'
            else:
                c = ' '
            header += c
        print(header)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                edge_mark = '-' if self.is_edge(i, j) else ' '
                row = f"   [{i}]{edge_mark}[{j}] :"
                for d in range(max_distance):
                    if self.vertex_distances[i][j][d] == MODE_I:
                        c = '█'
                    elif self.vertex_distances[i][j][d] == MODE_B:
                        c = '▒'
                    else:
                        c = ' '
                    row += c
                print(row)
        print()

    def draw(self, draw_disks: bool = False, ax=None):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if ax is None:
            _, ax = plt.subplots()

        # draw edges
        for u in range(self.n):
            for v in self.adj[u]:
                if u < v:  # avoid drawing twice
                    x_values = [self.coordinates[u].x, self.coordinates[v].x]
                    y_values = [self.coordinates[u].y, self.coordinates[v].y]
                    ax.plot(x_values, y_values, color="black", zorder=1)

        # draw vertices
        for i in range(self.n):
            ax.scatter(self.coordinates[i].x, self.coordinates[i].y,
                       color=colors[i % len(colors)], zorder=2)

        if draw_disks:
            for i in range(self.n):
                circle = plt.Circle(
                    (self.coordinates[i].x, self.coordinates[i].y),
                    self.unit,
                    color=colors[i % len(colors)],
                    fill=False,
                    linestyle="--",
                    zorder=0,
                )
                ax.add_patch(circle)

        ax.set_aspect("equal", adjustable="datalim")
        return ax

    def udg_recognition(self):
        self.start_time = time.time()
        self.last_verbose_time = self.start_time

        if not self.is_connected():
            self.stop_time = time.time()
            if self.verbose:
                print("Graph is not connected, cannot be a UDG.")
            return False

        if self.is_full():
            self.stop_time = time.time()
            if self.verbose:
                print("Graph is full, it is a UDG.")
            return True
        
        self.apply_order()

        while True:
            if self.verbose:
                print(f"Checking unit: {self.unit}")
            result = self.has_discrete_realization()
            if result == YES:
                self.stop_time = time.time()
                return True
            if result == NO:
                self.stop_time = time.time()
                return False
            if self.is_limit_achieved():
                self.stop_time = time.time()
                if self.verbose:
                    print("Reached max unit = {self.unit}, no realization found.")
                return False

            self.refine_granularity()
    
    def refine_granularity(self):
        self.set_unit(max(self.unit + 1, int(np.ceil(self.unit * 1.4))))

    def is_connected(self):
        """Check if the graph is connected using a simple BFS."""
        visited = [False] * self.n
        queue = deque([0])
        visited[0] = True
        while queue:
            v = queue.popleft()
            for neighbor in self.neighbors(v):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return all(visited)
    
    def is_full(self):
        """Check if the graph is a full."""
        for u in range(self.n):
            for v in range(u + 1, self.n):
                if not self.is_edge(u, v):
                    return False
        return True

    def is_limit_achieved(self):
        # temp
        self.unit > 2**self.n

    def has_discrete_realization(self):
        for only_I in [True, False]:
            if self.verbose:
                print(f"  {'Inner' if only_I else 'All'}")
            count_I: int = 0
            count_B: int = 0
            result = self.place_next_vertex(0, False, only_I, count_I, count_B)
            if result == YES:
                return YES
        return result
    
    def calculate_vertex_distances(self):
        self.result_I = 0
        self.result_B = 0
        self.vertex_distances = [[[MODE_O for _ in range(discrete_disk.RO[self.unit] * self.n)] for _ in range(self.n)] for _ in range(self.n)]

        self.apply_order()

        self.place_next_vertex(0, True, False, 0, 0)
    
    def store_vertex_distances(self, mode: np.uint8 = MODE_U):
        if self.verbose:
            msg = ""
            for i in range(self.n):
                coord = self.coordinates[i]
                msg += f"{i}:({coord.x:3d},{coord.y:3d}) "
            print(msg)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                coord_i = self.coordinates[i]
                coord_j = self.coordinates[j]
                distance = sqrt((coord_i.x - coord_j.x)**2 + (coord_i.y - coord_j.y)**2)
                distanceF = int(np.floor(distance))
                distanceC = int(np.ceil (distance))
                if self.vertex_distances[i][j][distanceF] != MODE_I:
                    self.vertex_distances[i][j][distanceF] = mode
                if self.vertex_distances[i][j][distanceC] != MODE_I:
                    self.vertex_distances[i][j][distanceC] = mode
    
    def place_next_vertex(self, j: int, calc_D: bool, only_I: bool, count_I: int, count_B: int):
        v_index = self.order[j]

        P = self.candidate_points(j, only_I, count_I, count_B)

        found_trigraph = False
        if self.verbose:
            self.set_iteration_len(v_index, len(P))
        point_iter = -1
        for p in P:
            incr_I = 1 if p.mode == MODE_I else 0
            incr_B = 1 if p.mode == MODE_B else 0

            self.set_coordinate(v_index, p.x, p.y, p.mode)
            self.clear_previous_area(j)

            if self.verbose:
                point_iter += 1
                self.set_iteration_index(v_index, point_iter)
                if time.time() - self.last_verbose_time > 10:
                    self.last_verbose_time = time.time()
                    print("  placing " + self.state_info(only_I, j))
            if j < self.n - 1:
                result = self.place_next_vertex(j + 1, calc_D, only_I, count_I + incr_I, count_B + incr_B)
                if result == YES:
                    return YES
                if result == TRIGRAPH:
                    if not only_I:
                        return TRIGRAPH
                    if not calc_D:
                        found_trigraph=True
            else:
                # if self.is_udg_realization():
                if count_I + incr_I == self.n:
                    if calc_D:
                        self.store_vertex_distances(MODE_I)
                        self.result_I += 1
                    else:
                        return YES
                if not only_I:
                    if calc_D:
                        self.store_vertex_distances(MODE_B)
                        self.result_B += 1
                    else:
                        return TRIGRAPH
                if not calc_D:
                    found_trigraph = True
        
        if not found_trigraph:
            return NO

        return TRIGRAPH
    
    def state_info(self, only_I: bool, j:int) -> str:
        info = f" {'I' if only_I else 'A'}"
        for i in range(j+1):
            k = self.order[i]
            c = self.coordinates[k]
            w = self.iteretions[k]
            x = c.x / self.unit
            y = c.y / self.unit
            info += f"  [{w.point_iter+1:4d}/{w.point_size:4d}] {k} {MODES[c.mode]} ({x: =6.3f}:{y: =6.3f})"
        return info
    
    def is_udg_realization(self) -> bool:
        return all(coord.mode == MODE_I for coord in self.coordinates)
    
    def candidate_points(self, j: int, only_I: bool, count_I: int, count_B: int) -> list[Coordinate]:
        if j == 0:
            P = []
            P.append(Coordinate(x = 0, y = 0, mode = MODE_I))
            return P
        if j == 1:
            area = DiscreteDisk.disk(self.unit, 0, 0, connected = True)
            if self.limit_points:
                P = [p for p in area.points_iter(types = ('I' if only_I else 'IB')) if p.y == 0 and p.x >= 0]
            else:
                P = [p for p in area.points_iter(types = ('I' if only_I else 'IB')) if p.y >= 0 and p.x >= 0]
            return P

        i = j - 2
        while i >= 0 and self.previous_area[j][i] is DISK_NONE:
            i -= 1

        neighbors_v_order_j = self.neighbors(self.order[j])

        for k in range(i+1, j):
            coord_v_order_k = self.coordinates[self.order[k]]
            area = DiscreteDisk.disk(self.unit, coord_v_order_k.x, coord_v_order_k.y, connected = self.order[k] in neighbors_v_order_j)
            if k > 0:
                prev_area = self.previous_area[j][k-1]
                area = create_area_by_join(prev_area, area) 
            self.previous_area[j][k] = area

        if j == 2:
            if self.limit_points:
                P = [p for p in area.points_iter(types = ('I' if only_I else 'IB')) if p.y >= 0]
                return P
            else:
                return area.points_list(types = ('I' if only_I else 'IB'))
        else: 
            return area.points_list(types = ('I' if only_I else 'IB'))

    def apply_order(self):
        """Choose and apply the appropriate ordering mode based on order_mode setting."""
        if self.order_mode == 'P':
            self.calculate_order_path()
        elif self.order_mode == 'DA':
            self.calculate_order_degree_level(desc = False)
        elif self.order_mode == 'DD':
            self.calculate_order_degree_level(desc = True)
        else:
            self.calculate_order_same()

    def calculate_order_same(self):
        self.order = range(self.n)

    def calculate_order_path(self):
        """Calculate a work order of the graph starting from any p3 inducted subgraph."""
        # Find a P3 (path of length 2) induced subgraph: vertices 0-1-2 such that
        # 0-1 and 1-2 are edges, but 0-2 is not an edge.
        for v0 in range(self.n):
            for v1 in self.neighbors(v0):
                for v2 in self.neighbors(v1):
                    if v2 == v0:
                        continue
                    if not self.is_edge(v0, v2):
                        # Found P3: v0-v1-v2
                        visited = [False] * self.n
                        self.order = [v0, v1, v2]
                        visited[v0] = visited[v1] = visited[v2] = True
                        q = deque([v0, v1, v2])
                        while q:
                            v = q.popleft()
                            for w in self.neighbors(v):
                                if not visited[w]:
                                    visited[w] = True
                                    self.order.append(w)
                                    q.append(w)

    def calculate_order_degree_level(self, desc: bool = True):
        """Calculate a work order starting from the highest degree vertex, then its neighbors (by degree), then their neighbors, etc., for any number of levels."""
        degrees = [len(self.adj[v]) for v in range(self.n)]
        used = [False] * self.n
        order = []

        # Start from the vertex with the highest degree
        current_level = [max(range(self.n), key=lambda v: degrees[v]) if desc else min(range(self.n), key=lambda v: degrees[v])]
        used[current_level[0]] = True
        order.append(current_level[0])

        while len(order) < self.n:
            next_level = set()
            for v in current_level:
                for u in self.neighbors(v):
                    if not used[u]:
                        next_level.add(u)
            # Remove already used vertices and sort by degree
            next_level = sorted(next_level, key=lambda v: degrees[v], reverse=desc)
            for v in next_level:
                if not used[v]:
                    order.append(v)
                    used[v] = True
            current_level = next_level
            # If no new vertices found, add any remaining unused vertices
            if not current_level and len(order) < self.n:
                remaining = [v for v in range(self.n) if not used[v]]
                remaining = sorted(remaining, key=lambda v: degrees[v], reverse=desc)
                for v in remaining:
                    order.append(v)
                    used[v] = True
                break

        self.order = order

def tests(verbose=False):
    # Example usage
    print("Test C3")
    g = Graph(3)
    g.set_verbose(verbose)
    g.add_edge(0,1)
    g.add_edge(1,2)
    g.add_edge(2,0)
    print("Graph C3 is UDG:", g.udg_recognition())

    print("Test P4")
    g = Graph(4)
    g.set_verbose(verbose)
    g.add_edge(0,1)
    g.add_edge(1,2)
    g.add_edge(2,3)
    print("Graph P4 is UDG:", g.udg_recognition())
    
    print("Test G5")
    g = Graph(7)
    g.set_verbose(verbose)
    g.add_edge(0,1)
    g.add_edge(0,3)
    g.add_edge(1,2)
    g.add_edge(1,4)
    g.add_edge(2,3)
    g.add_edge(2,5)
    g.add_edge(3,6)
    g.add_edge(4,5)
    g.add_edge(5,6)
    print("Graph G5 is non UDG:", g.udg_recognition())

    print("Test K2,3")
    g = Graph(5)
    g.set_verbose(verbose)
    g.add_edge(0,1)
    g.add_edge(0,2)
    g.add_edge(0,3)
    g.add_edge(1,4)
    g.add_edge(2,4)
    g.add_edge(3,4)
    print("Graph K2,3 is non UDG:", g.udg_recognition())

def test_coordinates_g4(verbose=False):
    g = Graph(7)
    g.set_verbose(verbose)
    g.set_coordinate(0,      0,      0)
    g.set_coordinate(1,  56568,      0)
    g.set_coordinate(2, -14142,  14142)
    g.set_coordinate(3,  35355,  63639)
    g.set_coordinate(4,  56568, -63639)
    g.set_coordinate(5, -63639, -35355)
    g.set_coordinate(6, - 7071, -70710)
    g.add_edge(0,1)
    g.add_edge(0,2)
    g.add_edge(1,3)
    g.add_edge(1,4)
    g.add_edge(2,3)
    g.add_edge(2,5)
    g.add_edge(4,6)
    g.add_edge(5,6)
    g.set_unit(70000)
    return g

def test_coordinates_g4a(verbose=False):
    a = 30
    b = 5
    sin_ap = sin((a+b) * pi / 180)
    cos_ap = cos((a+b) * pi / 180)
    sin_am = sin((a-b) * pi / 180)
    cos_am = cos((a-b) * pi / 180)
    u = 30000
    e = 1.05
    g = Graph(7)
    g.set_verbose(verbose)
    g.set_coordinate(0,                0,                0    )
    g.set_coordinate(1, - sin_ap     * u, - cos_ap     * u    )
    g.set_coordinate(2,                0, -              u * e)
    g.set_coordinate(3,   sin_ap     * u, - cos_ap     * u    )
    g.set_coordinate(4, - sin_am * 2 * u, - cos_am * 2 * u / e)
    g.set_coordinate(5,                0, -          2 * u * e)
    g.set_coordinate(6,   sin_am * 2 * u, - cos_am * 2 * u / e)
    g.add_edge(0,1)
    g.add_edge(0,3)
    g.add_edge(1,2)
    g.add_edge(1,4)
    g.add_edge(2,3)
    g.add_edge(3,6)
    g.add_edge(4,5)
    g.add_edge(5,6)
    g.set_unit(u)
    return g

def test_coordinates_g5(verbose=False):
    g = Graph(7)
    g.set_verbose(verbose)
    g.set_coordinate(0,     0,     0)
    g.set_coordinate(1, 28284,     0)
    g.set_coordinate(2, 28284, 14142)
    g.set_coordinate(3,  7071, 28284)
    g.set_coordinate(4, 56568,     0)
    g.set_coordinate(5, 49497, 28284)
    g.set_coordinate(6, 28284, 49497)
    g.add_edge(0,1)
    g.add_edge(0,3)
    g.add_edge(1,2)
    g.add_edge(1,4)
    g.add_edge(2,3)
    g.add_edge(2,5)
    g.add_edge(3,6)
    g.add_edge(4,5)
    g.add_edge(5,6)
    g.set_unit(30000)
    return g

def test_coordinates_g5a(verbose=False):
    a = 30
    sin_a = sin(a * pi / 180)
    cos_a = cos(a * pi / 180)
    u = 30000
    e = 0.578
    f = 1.154
    g = Graph(7)
    g.set_verbose(verbose)
    g.set_coordinate(0,               0,               0)
    g.set_coordinate(1,               0,           e * u)
    g.set_coordinate(2,   cos_a * e * u, - sin_a * e * u)
    g.set_coordinate(3, - cos_a * e * u, - sin_a * e * u)
    g.set_coordinate(4,   cos_a * f * u,   sin_a * f * u)
    g.set_coordinate(5,               0, -         f * u)
    g.set_coordinate(6, - cos_a * f * u,   sin_a * f * u)
    g.add_edge(0,1) 
    g.add_edge(0,2)
    g.add_edge(0,3)
    g.add_edge(1,4)
    g.add_edge(2,4)
    g.add_edge(2,5)
    g.add_edge(3,5)
    g.add_edge(3,6)
    g.add_edge(1,6)
    g.set_unit(u)
    return g

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check if a graph is a Unit Disk Graph (UDG).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-t", "--tests", action="store_true",
        help="Run example tests")
    group.add_argument(
        "-o", "--coordinates", action="store_true",
        help="Check graph with coordinates")
    group.add_argument(
        "-g", "--graph6", action="store_true",
        help="Check graph given as graph6")
    group.add_argument(
        "-e", "--edge_list", action="store_true",
        help="Check graph given as edge list")
    parser.add_argument(
        "-p", "--print_vertex", action="store_true",
        help="Print coordinates for each vertex")
    parser.add_argument(
        "-r", "--print_edges", action="store_true",
        help="Print distances for each edge")
    parser.add_argument(
        "-d", "--draw", action="store_true",
        help="Draw the graph using stored coordinates")
    parser.add_argument(
        "-c", "--circle", action="store_true",
        help="Draw unit disks when drawing the graph")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output for debugging")
    parser.add_argument(
        "-u", "--unit", type=int,
        help="Start unit")
    parser.add_argument(
        "-a", "--order", type=str, default="DD",
        help="Order mode")
    parser.add_argument(
        "-n", "--not_limit_points", action="store_true",
        help="Turn off limiting points")
    parser.add_argument(
        "graph", metavar="GRAPH", nargs="?", default="",
        help="Input graph description")

    args = parser.parse_args()

    check = False
    if args.tests:
        tests(args.verbose)
        return
    elif args.coordinates:
        if args.graph == 'g4':
            g = test_coordinates_g4(args.verbose)
        elif args.graph == 'g4a':
            g = test_coordinates_g4a(args.verbose)
        elif args.graph == 'g5':
            g = test_coordinates_g5(args.verbose)
        elif args.graph == 'g5a':
            g = test_coordinates_g5a(args.verbose)
    elif args.graph6:
        g = Graph(Graph6Converter.g6_to_graph(args.graph))
        check = True
    elif args.edge_list:
        g = Graph(Graph6Converter.edge_list_to_graph(args.graph))
        check = True

    g.set_verbose(args.verbose)

    if args.unit:
        g.set_unit(args.unit)

    if args.order:
        g.order_mode = args.order
        
    if args.not_limit_points:
        g.limit_points = False

    if check:
        output = g.udg_recognition()
        print("Graph is " + ("" if output else "NOT ") + "a Unit Disk Graph (UDG).")
    
    if args.print_vertex or args.print_edges:
        g.print_result(args.print_vertex, args.print_edges)

    if args.draw:
        g.draw(args.circle)
        import matplotlib.pyplot as plt
        plt.show()
        return

if __name__ == "__main__":
    main()
