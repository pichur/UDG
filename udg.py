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
from discrete_disk import DiscreteDisk

class Graph:
    """Simple adjacency list graph that can be built from an integer number
    of vertices or from a :class:`networkx.Graph` instance."""

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
        self.vertices = np.recarray(shape = self.n, dtype= [('x', 'int64'),('y', 'int64')])

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

    def add_edge(self, u, v):
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)

    def neighbors(self, v):
        return self.adj[v]

    def is_edge(self, u, v):
        return v in self.adj[u]
    
    def set_verbose(self, verbose):
        """Set verbosity for debugging purposes."""
        self.verbose = verbose
        return self
    
    def set_coordinate(self, v: int, x: int, y: int):
        """Set coordinates for vertex ``v``."""
        self.vertices[v].x = x
        self.vertices[v].y = y
        return self

    def set_unit(self, unit: int):
        """Set the unit disk radius for the graph."""
        self.unit = unit
        self.unit_sq = self.unit * self.unit
        #if hasattr(self, "eps"):
        #    self.apply_granularity()
        return self
    
    def vertex_distance_squared(self, u: int, v: int) -> int:
        """Return squared Euclidean distance between vertices ``u`` and ``v``."""
        dx = self.vertices[u].x - self.vertices[v].x
        dy = self.vertices[u].y - self.vertices[v].y
        return dx * dx + dy * dy

    def vertex_distance(self, u: int, v: int) -> float:
        """Return Euclidean distance between vertices ``u`` and ``v``."""
        return sqrt(self.vertex_distance_squared(u, v))

    def print_coordinates(self, print_vertex: bool, print_edges: bool) -> None:
        if (print_vertex):
            for i in range(self.n):
                print(f"  V ({i}): ({self.vertices[i].x:7d}, {self.vertices[i].y:7d})")
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

    def draw(self, draw_disks: bool = False, ax=None):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if ax is None:
            _, ax = plt.subplots()

        # draw edges
        for u in range(self.n):
            for v in self.adj[u]:
                if u < v:  # avoid drawing twice
                    x_values = [self.vertices[u].x, self.vertices[v].x]
                    y_values = [self.vertices[u].y, self.vertices[v].y]
                    ax.plot(x_values, y_values, color="black", zorder=1)

        # draw vertices
        for i in range(self.n):
            ax.scatter(self.vertices[i].x, self.vertices[i].y,
                       color=colors[i % len(colors)], zorder=2)

        if draw_disks:
            for i in range(self.n):
                circle = plt.Circle(
                    (self.vertices[i].x, self.vertices[i].y),
                    self.unit,
                    color=colors[i % len(colors)],
                    fill=False,
                    linestyle="--",
                    zorder=0,
                )
                ax.add_patch(circle)

        ax.set_aspect("equal", adjustable="datalim")
        return ax

    def udg_recognition(self, initial_epsilon=1):
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
        
        self.eps = initial_epsilon
        self.apply_granularity()

        while True:
            if self.verbose:
                print(f"Checking unit: {self.unit} epsilon: {self.eps}")
            result = self.has_discrete_realization()
            if result == YES:
                self.stop_time = time.time()
                return True
            if result == NO:
                self.stop_time = time.time()
                return False
            if self.min_eps():
                self.stop_time = time.time()
                if self.verbose:
                    print("Reached minimum epsilon, no realization found.")
                return False

            self.refine_granularity()
    
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
                if not self.has_edge(u, v):
                    return False
        return True

    def min_eps(self):
        """Check if the current epsilon is below the minimum threshold."""
        # temp
        return (self.eps << 10) > self.unit
    
    def has_discrete_realization(self):
        order = self.work_order()
        result = self.place_next_vertex(order, 0)
        return result

    def place_next_vertex(self, order, j: int):
        v = order[j]
        P = self.candidate_points(order, j)

        found_trigraph = False
        for p in P:
            self.set_coordinate(v, p[0], p[1])
            if self.verbose:
                if time.time() - self.last_verbose_time > 10:
                    self.last_verbose_time = time.time()
                    print("  placing " + self.state_info())
            if j < self.n - 1:
                result = self.place_next_vertex(order, j+1)
                if result == YES:
                    return YES
                if result == TRIGRAPH_ONLY:
                    found_trigraph=True
            else:
                found_trigraph = True
                if self.is_udg_realization():
                    return YES
        
        if not found_trigraph:
            return NO

        return TRIGRAPH_ONLY
    
    def state_info(self) -> str:
        return "TODO"
    
    def is_udg_realization(self) -> bool:
        print("TODO")
        return True
    
#     def refine_granularity(self):
#         self.eps = max(self.eps / 2, self.eps_min)
#         self.apply_granularity()

    def apply_granularity(self):
        self.eps_sqrt2 = self.eps * SQRT_2
        self.r_in      = self.unit - self.eps_sqrt2
        self.r_out     = self.unit + self.eps_sqrt2
        self.r_in_sq   = self.unit_sq - 2 * self.eps_sqrt2 * self.unit + 2 * self.eps
        self.r_out_sq  = self.unit_sq + 2 * self.eps_sqrt2 * self.unit + 2 * self.eps

    def candidate_points(self, order, j: int):
        P = []
        if j == 0:
            P.append((0, 0))
            return P
        if j == 1:
            for x in range(0, discrete_disk.RO[self.unit]):
                P.append((x, 0))
            return P
        if j == 2:
            v1 = self.vertices[order[1]]
            dd = DiscreteDisk.disk(self.unit, x = v1.x, y = v1.y) # connected to previous (1) vertex
            dd.disconnect(r=self.unit, x = 0, y = 0) # disconnected from fisrt (0) vertex
            P = [p for p in dd.points_iter() if p[1] >= 0]
            return P

        v = order[j]
        neighs    = [order[k] for k in range(j) if order[k]     in self.neighbors(v)]
        nonneighs = [order[k] for k in range(j) if order[k] not in self.neighbors(v)]

        if not neighs:
            raise ValueError("Missing neighborhoot for vertex " + v)
        
        it = iter(neighs)
        first = next(it)
        v = order[first]
        dd = DiscreteDisk.disk(self.unit, x = v.x, y = v.y)
        for i in it:
            v = order[i]
            dd.connect(self.unit, x = v.x, y = v.y)

        for i in nonneighs:
            v = order[i]
            dd.disconnect(self.unit, x = v.x, y = v.y)

        return dd.points_list

    def work_order(self):
        """Return a work order of the graph starting from any p3 inducted subgraph."""
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
                        order = [v0, v1, v2]
                        visited[v0] = visited[v1] = visited[v2] = True
                        q = deque([v0, v1, v2])
                        while q:
                            v = q.popleft()
                            for w in self.neighbors(v):
                                if not visited[w]:
                                    visited[w] = True
                                    order.append(w)
                                    q.append(w)
                        return order

#     def dist_gte_r_out(self, p1, p2):
#         dp2 = dist_p2(p1,p2)
#         return dp2 >= self.r_out_p2
    
#     def dist_lte_r_in(self, p1, p2):
#         dp2 = dist_p2(p1,p2)
#         return dp2 <= self.r_in_p2

#     def is_udg_realization(self, coords):
#         has_optional_edge_for_mandatory = False
#         has_optional_edge_for_forbidden = False

#         for i in range(self.n):
#             for j in range (i+1, self.n):
#                 dp2 = dist_p2(coords[i], coords[j])
                
#                 if ((dp2 > self.r_in_p2) and
#                     (dp2 <= self.r_out_p2)):
#                     if self.is_edge(i, j):
#                         has_optional_edge_for_mandatory = True
#                     else:
#                         has_optional_edge_for_forbidden = True

#         return (not has_optional_edge_for_mandatory) or (not has_optional_edge_for_forbidden)
    
# def dist_p2(p1, p2):
#     return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

SQRT_2 = sqrt(2)

TRIGRAPH_ONLY = "TRIGRAPH_ONLY"
YES = "YES"
NO = "NO"

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
        g.print_coordinates(args.print_vertex, args.print_edges)
    elif args.graph6:
        g = Graph(Graph6Converter.g6_to_graph(args.graph))
        check = True
    elif args.edge_list:
        g = Graph(Graph6Converter.edge_list_to_graph(args.graph))
        check = True

    g.set_verbose(args.verbose)

    if check:
        output = g.udg_recognition()
        print("Graph is " + ("" if output else "NOT ") + "a Unit Disk Graph (UDG).")

    if args.draw:
        g.draw(args.circle)
        import matplotlib.pyplot as plt
        plt.show()
        return

if __name__ == "__main__":
    main()

