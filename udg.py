"""Simple Unit Disk Graph recognition algorithm."""

from math import sqrt
from collections import deque
from typing import ClassVar
import networkx as nx
import argparse
import time
import random
import Graph6Converter
import numpy as np
import matplotlib.pyplot as plt
import discrete_disk
from ug import GraphUtil
from discrete_disk import create_area_by_join, DiscreteDisk, Coordinate, MODES, MODE_U, MODE_I, MODE_B, MODE_O, DISK_NONE
from dataclasses import dataclass


TRIGRAPH = "TRIGRAPH"
YES = "YES"
NO = "NO"
# Log levels, 0 OFF, 1 BASIC, increasing levels of verbosity, negative for special modes, -1 for work order
LOG_OFF   = 0
LOG_BASIC = 1
LOG_INFO  = 2
LOG_DEBUG = 3
LOG_TRACE = 4
LOG_WORK_ORDER = -1

@dataclass(slots=True)
class IterationInfo:
    point_iter: int
    point_size: int

class Graph:
    """Simple adjacency list graph that can be built from an integer number
    of vertices or from a :class:`networkx.Graph` instance."""
    log_level: int # 0 OFF, 1 BASIC, increasing levels of verbosity, negative for special modes, -1 for work order
    print_progress: int = 0
    limit_points:bool = True
    limit_negative_distances: bool = False
    optimize_for_yes: bool = False
    forbid_same_positions: bool = False

    level: int = 0
    unit: int
    max_unit: int = 0
    n: int
    adj: list[list[int]]

    order_mode: str = 'DD'
    point_iteration_order: str = 'none'

    order: list[int]
    coordinates: list[Coordinate]

    previous_area: list[list[any]]

    iterations: list[IterationInfo]

    check_distance = False
    check_distance_iteration = 0
    node_distances: list[int] = []

    collect_work_summary: bool = False
    place_next_vertex_counter: int = 0
    place_next_vertex_by_level_and_order_counter: list[list[int]] = []

    stop_time = False
    
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
        
        self.log_level = LOG_BASIC

        # store vertex coordinates and additional parameters
        self.coordinates = [
            Coordinate(np.int64(0), np.int64(0), MODE_U)
            for _ in range(self.n)
        ]

        self.iterations = [
            IterationInfo(0, 0)
            for _ in range(self.n)
        ]

        # set to real values during processing
        # indexing by permuted work order
        self.previous_area = [[False for _ in range(self.n)] for _ in range(self.n)]

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

        self.calculate_vertex_edge_distance()

    def add_edge(self, u: int, v: int):
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)

    def neighbors(self, v: int):
        return self.adj[v]

    def is_edge(self, u: int, v: int):
        return v in self.adj[u]
    
    def set_log_level(self, log_level):
        """Set verbosity for debugging purposes."""
        self.log_level = log_level
        return self
    
    def get_place_next_vertex_counter(self) -> int:
        return self.place_next_vertex_counter

    def reset_place_next_vertex_counter(self) -> None:
        self.place_next_vertex_counter = 0

    def set_collect_work_summary(self, collect_work_summary: bool):
        """Set whether to collect work summary statistics."""
        self.collect_work_summary = collect_work_summary
        return self

    def set_print_progress(self, print_progress: int):
        """Set the interval in seconds for printing progress information."""
        self.print_progress = print_progress
        return self

    def set_coordinate(self, v: int, x: int, y: int, mode: np.uint8 = MODE_U):
        """Set coordinates for vertex ``v``."""
        v = self.coordinates[v]
        v.x, v.y, v.mode = x, y, mode
        return self
    
    def clear_previous_area(self, order_index: int):
        for row in self.previous_area: row[order_index] = False

    def set_iteration_len(self, v: int, len: int):
        c = self.iterations[v].point_size = len
        return self

    def set_iteration_index(self, v: int, index: int):
        c = self.iterations[v].point_iter = index
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
    
    def calculate_vertex_edge_distance(self):
        """Calculate shortest path distances between all pairs of vertices."""
        # Initialize with infinity for all pairs
        INF = float('inf')
        self.vertex_edge_distance = [[INF] * self.n for _ in range(self.n)]
        
        # Distance from vertex to itself is 0
        for i in range(self.n):
            self.vertex_edge_distance[i][i] = 0

        # Set direct edge distances to 1
        for u in range(self.n):
            for v in self.adj[u]:
                self.vertex_edge_distance[u][v] = 1
        
        # Floyd-Warshall algorithm
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if (self.vertex_edge_distance[i][k] != INF and 
                        self.vertex_edge_distance[k][j] != INF and
                        self.vertex_edge_distance[i][k] + self.vertex_edge_distance[k][j] < self.vertex_edge_distance[i][j]):
                        self.vertex_edge_distance[i][j] = self.vertex_edge_distance[i][k] + self.vertex_edge_distance[k][j]

    def print_result(self, print_vertex: bool, print_edges: bool, print_work_summary: bool) -> None:
        if self.stop_time:
            time = self.stop_time - self.start_time
            if self.log_level >= LOG_BASIC:
                print(f"Time consumed: {(int)(1000*time)} ms")
        if (print_work_summary):
            print(f"Order: {self.order}")
            print(f"Total place_next_vertex calls: {self.get_place_next_vertex_counter()}")
            print(f"  lvl:ord  calls")
            for level, counters in enumerate(self.place_next_vertex_by_level_and_order_counter):
                for order, count in enumerate(counters):
                    print(f"  {level:3d}:{order:3d}  {count:9,d}".replace(',', ' '))
            print(f"Total operation_disk calls: {DiscreteDisk.get_operation_disk_counter()}")
            print(f"Final unit: {self.unit}")

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

    def get_node_distances_info(self, header: bool = False, chars:str="█▒ "):
        maximum_vertex_edge_distance = self.get_maximum_vertex_edge_distance()
        maximum_vertex_distance = maximum_vertex_edge_distance * self.unit + 1

        if header:
            row_header = '>'
            for d in range(1, maximum_vertex_distance):
                if d % self.unit == 0:
                    c = '|'
                elif d % 10 == 0:
                    c = '*'
                elif d % 5 == 0:
                    c = '.'
                else:
                    c = ' '
                row_header += c

            row_header += '<'
            return row_header

        row = ""
        for d in range(maximum_vertex_distance+1):
            if d < len(self.node_distances):
                if self.node_distances[d] == MODE_I:
                    c = chars[0] if len(chars) > 0 else '█'
                elif self.node_distances[d] == MODE_B:
                    c = chars[1] if len(chars) > 1 else '▒'
                elif self.node_distances[d] == MODE_O:
                    c = chars[2] if len(chars) > 2 else ' '
                else:
                    c = '?'
            else:
                c = '-'
            row += c

        return row

    def get_maximum_vertex_edge_distance(self):
        max_distance = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                edge_distance = self.vertex_edge_distance[i][j]
                if edge_distance != float('inf'):
                    max_distance = max(max_distance, edge_distance)
        return max_distance

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
        self.last_progress_time = self.start_time

        if not self.is_connected():
            self.stop_time = time.time()
            if self.log_level >= LOG_BASIC:
                print("Graph is not connected, cannot be a UDG.")
            return False

        if self.is_full():
            self.stop_time = time.time()
            if self.log_level >= LOG_BASIC:
                print("Graph is full, it is a UDG.")
            return True
        
        self.apply_order()

        self.level = 0
        while True:
            if self.log_level >= LOG_BASIC:
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
                if self.log_level >= LOG_BASIC:
                    print("Reached max unit = {self.unit}, no realization found.")
                return False

            self.refine_granularity()
            self.level += 1
    
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
        if self.max_unit > 0:
            if self.unit > self.max_unit:
                if self.log_level >= LOG_BASIC:
                    print(f"Reached maximum unit limit: {self.max_unit}")
                return True
        # temp
        if self.unit > 2**self.n:
            if self.log_level >= LOG_BASIC:
                print(f"Reached theoretical unit limit: {2**self.n}")
            return True
        
        return False

    def has_discrete_realization(self):
        range_modes = [True] if self.optimize_for_yes else [True, False]
        for only_I in range_modes:
            if self.log_level >= LOG_INFO:
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

    def calculate_node_distances(self, nodes: tuple[int, int], order: str = "DD") -> list[int]:
        u, v = nodes
        if self.log_level >= LOG_INFO:
            print(f"Calculating node distances for: {u}, {v}")

        maximum_vertex_edge_distance = self.get_maximum_vertex_edge_distance()

        # +1 for 0 +1 for max
        self.node_distances = [MODE_O] * (maximum_vertex_edge_distance * self.unit + 2)

        self.apply_order(order, nodes)

        for only_I in [True, False]:
            self.check_distance = True
            self.check_distance_iteration = 0
            while self.check_distance:

                # To keep correct order of point not check only_I
                count_I: int = 0
                count_B: int = 0
                r_text = self.place_next_vertex(0, False, only_I, count_I, count_B)
                r_mode = MODE_U
                if r_text == YES:
                    r_mode = MODE_I
                elif r_text == NO:
                    r_mode = MODE_O
                elif r_text == TRIGRAPH:
                    r_mode = MODE_B
                else:
                    r_mode = MODE_U

                dx = self.coordinates[v].x

                if dx >= len(self.node_distances):
                    self.node_distances.extend([MODE_U] * (dx - len(self.node_distances) + 1))

                if self.node_distances[dx] != MODE_I:
                    self.node_distances[dx] = r_mode

                self.check_distance_iteration += 1

        return self.node_distances

    def store_vertex_distances(self, mode: np.uint8 = MODE_U):
        if self.log_level >= LOG_DEBUG:
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
    
    def mark_place_next_vertex_process(self, order_index: int):
        while len(self.place_next_vertex_by_level_and_order_counter) <= self.level:
            self.place_next_vertex_by_level_and_order_counter.append([])
        while len(self.place_next_vertex_by_level_and_order_counter[self.level]) <= order_index:
            self.place_next_vertex_by_level_and_order_counter[self.level].append(0)
        self.place_next_vertex_by_level_and_order_counter[self.level][order_index] += 1

    def minimize_coordinates(self, j: int, only_I: bool, count_I: int, count_B: int) -> list[Coordinate]:
        # Find the unplaced vertex with the smallest number of candidate points
        min_candidates = float('inf')
        best_vertex = None
        best_P = None

        for v in range(self.n):
            # Skip vertices that are already placed in order
            if v in self.order[:j]:
                continue
            
            # Temporarily set this vertex as the next one to get candidate points
            self.order[j] = v
            
            # Get candidate points for this vertex
            P = self.candidate_points(j, only_I, count_I, count_B)
            num_candidates = len(P)
            
            if self.log_level > LOG_TRACE:
                print(f"  check order {j}={v} : {num_candidates}")

            # Check if this is the best option so far
            if num_candidates < min_candidates:
                min_candidates = num_candidates
                best_vertex = v
                best_P = P

        # Set the best vertex at position j and use its candidate points
        self.order[j] = best_vertex
        P = best_P

    def place_next_vertex(self, j: int, calc_D: bool, only_I: bool, count_I: int, count_B: int):
        if self.order[j] == -1:
            P = self.minimize_coordinates(j, only_I, count_I, count_B)
        else:
            P = self.candidate_points(j, only_I, count_I, count_B)

        vertex = self.order[j]

        if self.log_level >= LOG_TRACE:
            print(f"order[{j}]={vertex} : {len(P)} points")

        found_trigraph = False

        if self.print_progress > 0:
            self.set_iteration_len(vertex, len(P))
        
        iter_p = -1
        for p in P:
            iter_p += 1
            self.set_iteration_index(vertex, iter_p)

            incr_I = 1 if p.mode == MODE_I else 0
            incr_B = 1 if p.mode == MODE_B else 0

            self.set_coordinate(vertex, p.x, p.y, p.mode)
            self.clear_previous_area(j)

            self.place_next_vertex_counter += 1
            if self.collect_work_summary:
                self.mark_place_next_vertex_process(j)
            
            if (self.print_progress > 0) and (time.time() - self.last_progress_time > self.print_progress):
                self.last_progress_time = time.time()
                if time.time() - self.last_verbose_time > 10:
                    self.last_verbose_time = time.time()
                    print("  placing " + self.state_info(only_I, j))
            if (self.log_level >= LOG_DEBUG) or (self.log_level == LOG_WORK_ORDER):
                print(f"vertex = {self.order[j]} already_placed = {j} coordinates = {self.print_coordinates(j)}")
            if j < self.n - 1:
                result = self.place_next_vertex(j + 1, calc_D, only_I, count_I + incr_I, count_B + incr_B)
                if result == YES:
                    return YES
                if result == TRIGRAPH:
                    if not only_I:
                        return TRIGRAPH
                    if not calc_D:
                        found_trigraph=True
                if not only_I and (result == NO):
                    # TODO print(f"XXX {self.unit} {j} forbidden at {self.state_info(only_I, j)}")
                    pass
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
            return TRIGRAPH if self.optimize_for_yes else NO

        return TRIGRAPH
    
    def print_coordinates(self, placed: int = -1) -> str:
        """Format coordinates in the requested format: [( 0, 0),(+7,-2),(-3,+5),(+3,+4),(+4, 0),(-1,+4),  None]"""
        coords = []
        for i in range(self.n): coords.append("     None")
        if placed == -1:
            placed = self.n - 1
        for i in range(placed + 1):
            vertex = self.order[i]
            coord = self.coordinates[vertex]
            x = coord.x
            y = coord.y
            # Format with proper signs and 2-digit width
            x_str = f"{x:+3d}" if x != 0 else "  0"
            y_str = f"{y:+3d}" if y != 0 else "  0"
            coords[vertex] = f"({x_str},{y_str})"
        
        return "[" + ",".join(coords) + "]"
    
    def state_info(self, only_I: bool, j:int) -> str:
        info = f" {'I' if only_I else 'A'}"
        for i in range(j+1):
            k = self.order[i]
            c = self.coordinates[k]
            s = self.iterations[k].point_size
            i = self.iterations[k].point_iter+1
            x = c.x
            y = c.y
            info += f"  [{i:4d}/{s:4d}] {k} {MODES[c.mode]} ({x: =3d}:{y: =3d})"
        return info
    
    def is_udg_realization(self) -> bool:
        return all(coord.mode == MODE_I for coord in self.coordinates)
    
    def candidate_points(self, j: int, only_I: bool, count_I: int, count_B: int) -> list[Coordinate]:
        P = self.candidate_points_without_order(j, only_I, count_I, count_B)

        if self.point_iteration_order == 'ascending':
            P.sort(key=lambda point: (point.y, point.x))
        elif self.point_iteration_order == 'descending':
            P.sort(key=lambda point: (-point.y, -point.x))
        elif self.point_iteration_order == 'spiral':
            P.sort(key=lambda point: (point.x * point.x + point.y * point.y, point.y, point.x))
        elif self.point_iteration_order == 'reverse_spiral':
            P.sort(key=lambda point: (-(point.x * point.x + point.y * point.y), -point.y, -point.x))
        elif self.point_iteration_order == 'zigzag':
            P.sort(key=lambda point: (point.y + point.x, point.y - point.x))
        elif self.point_iteration_order == 'reverse_zigzag':
            P.sort(key=lambda point: (-(point.y + point.x), -(point.y - point.x)))
        elif self.point_iteration_order == 'bisection':
            P.sort(key=lambda point: (abs(point.x - point.y), point.y, point.x))
        elif self.point_iteration_order == 'random':
            random.shuffle(P)
        # else: keep original order

        return P

    def remove_used_coordinates(self, j, P):      
        unique_P = []
        placed_positions = set()
        for i in range(j):
            vertex = self.order[i]
            coord = self.coordinates[vertex]
            placed_positions.add((coord.x, coord.y))
        
        for point in P:
            pos = (point.x, point.y)
            if pos not in placed_positions:
                unique_P.append(point)

        return unique_P
    
    def candidate_points_without_order(self, j: int, only_I: bool, count_I: int, count_B: int) -> list[Coordinate]:
        if j == 0:
            P = []
            P.append(Coordinate(x = 0, y = 0, mode = MODE_I))
            return P
        if j == 1:
            """ Previous is 0, so coordinate is equal (0,0) """
            area = self.create_area_for_next_vertex_join(0, 0, self.order[0], self.order[1], True)
            if self.limit_points:
                P = [p for p in area.points_iter(types = ('I' if only_I else 'IB')) if p.y == 0 and p.x >= 0]
            else:
                P = [p for p in area.points_iter(types = ('I' if only_I else 'IB')) if p.y >= 0 and p.x >= 0]

            if self.check_distance:
                if self.check_distance_iteration >= len(P) - 1:
                    # This is the last index, handle accordingly
                    p = P[-1]  # Get the last element
                    self.check_distance = False  # Stop further iterations
                else:
                    p = P[self.check_distance_iteration]
                P = []
                P.append(p)
            
            return P

        i = j - 2
        while i >= 0 and self.previous_area[j][i] is False:
            i -= 1
        
        for k in range(i+1, j):
            coord_v_order_k = self.coordinates[self.order[k]]
            area = self.create_area_for_next_vertex_join(coord_v_order_k.x, coord_v_order_k.y, self.order[j], self.order[k])
            if k > 0:
                prev_area = self.previous_area[j][k-1]
                area = create_area_by_join(prev_area, area) 
            self.previous_area[j][k] = area

        """For limit points return only positive y coordinates for second vertex"""
        if j == 2:
            if self.limit_points:
                P = [p for p in area.points_iter(types = ('I' if only_I else 'IB')) if p.y >= 0]
                return P
            else:
                return area.points_list(types = ('I' if only_I else 'IB'))
        else: 
            return area.points_list(types = ('I' if only_I else 'IB'))

    def create_area_for_next_vertex_join(self, x:int, y:int, u: int, v: int, force_limit_negative_distance: bool = False) -> DiscreteDisk:
        distance = self.vertex_edge_distance[u][v]
        if distance == 1:
            area = DiscreteDisk.disk(self.unit, x, y, connected = True)          
        else:
            area = DiscreteDisk.disk(self.unit, x, y, connected = False)
            if force_limit_negative_distance or self.limit_negative_distances:
                area = create_area_by_join(area, DiscreteDisk.disk(self.unit * distance, x, y, connected = True))
        return area


    def apply_order(self, order_mode: str = None, force_nodes: list[int] = None):
        """Choose and apply the appropriate ordering mode based on order_mode setting."""
        if order_mode is not None:
            self.order_mode = order_mode

        if self.order_mode.startswith('F'):
            self.calculate_order_by_forced_nodes(force_nodes)
        elif self.order_mode.startswith('P'):
            self.calculate_order_path()
        elif self.order_mode.startswith('DA'):
            self.calculate_order_degree_level(desc = False)
        elif self.order_mode.startswith('DD'):
            self.calculate_order_degree_level(desc = True)
        elif self.order_mode[0].isdigit():
            self.order = [-1] * self.n
            # Custom order provided as a suffix
            order_str = self.order_mode
            order_str_list = order_str.split(',')
            order_int_list = []
            for x in order_str_list:
                x = x.strip()
                if x.isdigit():
                    order_int_list.append(int(x))
            # Find the maximum vertex value to determine if indexed from 0 or 1
            max_vertex = -1
            for i in range(min(self.n, len(order_int_list))):
                max_vertex = max(max_vertex, order_int_list[i])
            
            # Determine offset: if max_vertex >= n, assume 1-based indexing
            offset = 0 if max_vertex < self.n else max_vertex - self.n + 1
            
            if offset > 0:
                for i in range(min(self.n, len(order_int_list))):
                    if order_int_list[i] >= offset:
                        order_int_list[i] = order_int_list[i] - offset
            for i in range(min(self.n, len(order_int_list))):
                self.order[i] = order_int_list[i]
        else:
            self.calculate_order_same()
        
        if force_nodes is not None:
            # Ensure forced nodes are at the start of the order
            forced_set = set(force_nodes)
            remaining_nodes = [v for v in self.order if v not in forced_set]
            self.order = list(force_nodes) + remaining_nodes

        if self.order_mode.__contains__('auto'):
            # Keep first two nodes and set rest to -1
            if len(self.order) >= 2:
                self.order = list(self.order[:2]) + [-1] * (len(self.order) - 2)
        
        if self.log_level >= LOG_INFO:
            print(f"Using order mode: {self.order_mode}")
            print(f"Work order: {self.order}")
    
    def calculate_order_by_forced_nodes(self, force_nodes: list[int]):
        """Calculate a work order of the graph starting from the given forced nodes."""
        visited = [False] * self.n
        self.order = []
        q = deque()

        for v in force_nodes:
            if not visited[v]:
                visited[v] = True
                self.order.append(v)
                q.append(v)

        while q:
            v = q.popleft()
            for w in self.neighbors(v):
                if not visited[w]:
                    visited[w] = True
                    self.order.append(w)
                    q.append(w)

        # Add any remaining unvisited nodes
        for v in range(self.n):
            if not visited[v]:
                visited[v] = True
                self.order.append(v)
                
    def calculate_order_same(self):
        self.order = []
        for v in range(self.n):
            self.order.append(v)
            
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

def write_out(output_file, msg_info):
    if output_file:
        with open(output_file, 'a') as out_file:
            out_file.write(msg_info)

def main() -> None:
    import graph_examples
    # abcdefghijklmnopqrstuvwxyz
    # -b-d----ij------qr-------z
    parser = argparse.ArgumentParser(
        description="Check if a graph is a Unit Disk Graph (UDG).")
    # main
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-t", "--tests", action="store_true",
        help="Run example tests")
    group.add_argument(
        "-c", "--coordinates", action="store_true",
        help="Check graph with coordinates")
    group.add_argument(
        "-g", "--graph6", action="store_true",
        help="Check graph given as graph6")
    group.add_argument(
        "-e", "--edge_list", action="store_true",
        help="Check graph given as edge list")
    parser.add_argument(
        "-f", "--file", action="store_true",
        help="Read graphs from file (one graph6 per line)")
    parser.add_argument(
        "graph", metavar="GRAPH", nargs="?", default="",
        help="Input graph description (multiple graphs can be separated by ;) or file if -f is used)")
    parser.add_argument(
        "-u", "--unit", type=int, default=4,
        help="Start unit")
    parser.add_argument(
        "-x", "--max_unit", type=int, default=0,
        help="Maximum unit, 0 for real max limit")
    parser.add_argument(
        "-o", "--order", type=str, default="DD",
        help="Order mode (F:forced nodes (given in parameter);" \
                         "P:path;" \
                         "DA:degree ascending;" \
                         "DD:degree descending;" \
                         "v1,...,vn:custom order - starting from digit;" \
                         "ALL - iterate over all permutations;" \
                         "other:same order); default DD")
    # settings
    parser.add_argument(
        "-n", "--not_limit_points", action="store_true",
        help="Turn off limiting points")
    parser.add_argument(
        "-m", "--limit_negative_distances", action="store_true",
        help="Limit negative distances")
    parser.add_argument(
        "-y", "--optimize_for_yes", action="store_true",
        help="Check only for UDG realization, not for missing trigraphs, stop by limit only")
    parser.add_argument(
        "-k", "--point_iteration_order", type=str, default="none",
        help="Point iteration order: none (default), ascending, descending, random")
    parser.add_argument(
        "-s", "--allow_same_positions", action="store_true",
        help="Allow same positions for different vertices, default false caused by auto detection of same vertex in graph")
    # processing
    parser.add_argument(
        "-a", "--preprocess", type=str, default="rs",
        help="Preprocess steps, can contains: r - reduce; s - check for known non-udg subgraphs")
    # output
    parser.add_argument(
        "-v", "--verbose", type=str, default="",
        help="Verbose modes, can contains: e - edge list; c - vertex coordinates; w - work summary; d - draw graph; u - draw unit disks")
    parser.add_argument(
        "-p", "--print_progress", type=int, default=0,
        help="Print progress information every n seconds, 0 to turn off")
    parser.add_argument(
        "-l", "--log_level", type=int, default=1,
        help="Log level, 0 OFF, 1 BASIC, 2 INFO, 3 DEBUG, 4 TRACE, negative for special modes, -1 for work order; default BASIC")
    parser.add_argument(
        "-w", "--output_file", type=str,
        help="Write output to file")

    args = parser.parse_args()

    check = False
    if args.tests:
        graph_examples.tests(args.log_level)
        return
    
    graph_K23 = nx.Graph()
    graph_K23.add_edges_from([
        (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4)  
    ])
    graph_S6 = nx.Graph()
    graph_S6.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)])

    # Parse verbose modes from the verbose string
    print_edges              = 'e' in args.verbose
    print_vertex_coordinates = 'c' in args.verbose  
    print_work_summary       = 'w' in args.verbose
    draw_graph               = 'd' in args.verbose
    draw_unit_circle         = 'u' in args.verbose

    # Parse preprocess steps from the preprocess string
    preprocess_reduce_graph             = 'r' in args.preprocess
    preprocess_check_non_udg_subgraphs  = 's' in args.preprocess

    if args.allow_same_positions:
        DiscreteDisk.allow_same_positions = True
    else:
        DiscreteDisk.allow_same_positions = False

    if args.file:
        with open(args.graph, 'r') as f:
            graphs_input = [line.strip() for line in f if line.strip()]
    else:
        # Split input by semicolons for multiple graphs
        if ';' in args.graph:
            graphs_input = [g.strip() for g in args.graph.split(';') if g.strip()]
        else:
            graphs_input = [args.graph] if args.graph else []

    if not graphs_input:
        print("No graph provided")
        return

    for gi_i, graph_input in enumerate(graphs_input):
        nx_g = None
        if args.coordinates:
            g = graph_examples.get_test_graph_by_name(graph_input, args.log_level)
        elif args.graph6:
            nx_g = Graph6Converter.g6_to_graph(graph_input)
            g = Graph(nx_g)
            check = True
        elif args.edge_list:
            nx_g = Graph6Converter.edge_list_to_graph(graph_input)
            g = Graph(nx_g)
            check = True

        node_orders = []
        if args.order == "ALL":
            from itertools import permutations
            vertices = list(range(g.n))
            all_permutations = list(permutations(vertices))
            for perm in all_permutations:
                node_orders.append(','.join(str(vertex) for vertex in perm))
        else:
            node_orders.append(args.order)
        
        for no_i, node_order in enumerate(node_orders):
            if (no_i > 0) and nx_g is not None:
                g = Graph(nx_g)

            g.set_collect_work_summary(print_work_summary)

            g.set_log_level     (args.log_level     )
            g.set_print_progress(args.print_progress)

            if args.unit:
                g.set_unit(args.unit)

            g.max_unit = args.max_unit

            g.order_mode = node_order
            
            if args.point_iteration_order: g.point_iteration_order = args.point_iteration_order

            if args.not_limit_points        : g.limit_points             = False
            if args.limit_negative_distances: g.limit_negative_distances = True
            if args.optimize_for_yes        : g.optimize_for_yes         = True

            msg_info = ""
            if len(graphs_input) > 1:
                msg_info = f"Graph ({(gi_i+1):4d}/{len(graphs_input):4d}) {graph_input} : "
            if len(node_orders) > 1:
                msg_info += f" Order ({(no_i+1):4d}/{len(node_orders):4d}) {node_order} : "
            write_out(args.output_file, msg_info)
            stop_check = False
            if check:
                if preprocess_reduce_graph:
                    reduction_info = GraphUtil.reduce(nx_g)
                    if reduction_info.reduced_nodes > 0:
                        stop_check = True
                        msg_break = f" reduced to graph {reduction_info.output_canonical_g6}; reduced nodes {reduction_info.reduced_nodes}"
                elif preprocess_check_non_udg_subgraphs:
                    if GraphUtil.contains_induced_subgraph(nx_g, graph_K23):
                        stop_check = True
                        msg_break += " contains K2,3 induced subgraph"
                    elif GraphUtil.contains_induced_subgraph(nx_g, graph_S6):
                        stop_check = True
                        msg_break += " contains S6 induced subgraph"
                
                if stop_check:
                    write_out(args.output_file, msg_break + '\n')
                    msg_info += msg_break
                else:
                    start_time = time.time()
                    udg_check_result = g.udg_recognition()
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    msg_stop = "   UDG  " if udg_check_result else " non udg"
                    msg_stop += f" time = {int(1000*elapsed_time):9d} ms pnv={g.place_next_vertex_counter:12d} do={DiscreteDisk.get_operation_disk_counter():12d} unit={g.unit:2d} " + g.print_coordinates()
                    write_out(args.output_file, msg_stop + '\n')
                    msg_info += msg_stop

                print(msg_info)

                g.print_result(print_vertex_coordinates, print_edges, print_work_summary)

            if draw_graph:
                g.draw(draw_unit_circle)
                import matplotlib.pyplot as plt
                plt.show()
                return

if __name__ == "__main__":
    main()
