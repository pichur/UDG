"""Simple Unit Disk Graph recognition algorithm."""

from math import sqrt
from collections import deque
import networkx as nx
import argparse
import time
import Graph6Converter
import numpy as np

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
        self.unit_squared = self.unit * self.unit
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

    def show_all_distances(self) -> None:
        """Print distances for all vertex pairs with comparison to unit range."""
        for i in range(self.n):
            for j in range(i + 1, self.n):
                e = self.is_edge(i, j)
                edisp = "E" if e else " "
                d_squared = self.vertex_distance_squared(i, j)
                comp = "<=" if d_squared <= self.unit_squared else "> "
                d = sqrt(d_squared)
                print(f"  {edisp} dist({i},{j}) = {comp} {d:.3f}")

    def draw(self, draw_disks: bool = False, ax=None):
        """Visualize the graph using stored vertex coordinates.

        Parameters
        ----------
        draw_disks : bool, optional
            If ``True`` draw a circle of radius ``self.unit`` around every
            vertex to visualise the unit disk.  Defaults to ``False``.
        ax : :class:`matplotlib.axes.Axes`, optional
            Existing axes to draw on.  When ``None`` a new figure and axes
            are created.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object the graph was drawn on.
        """
        import matplotlib.pyplot as plt

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
        xs = [self.vertices[i].x for i in range(self.n)]
        ys = [self.vertices[i].y for i in range(self.n)]
        ax.scatter(xs, ys, color="red", zorder=2)

        if draw_disks:
            for i in range(self.n):
                circle = plt.Circle(
                    (self.vertices[i].x, self.vertices[i].y),
                    self.unit,
                    color="blue",
                    fill=False,
                    linestyle="--",
                    zorder=0,
                )
                ax.add_patch(circle)

        ax.set_aspect("equal", adjustable="datalim")
        return ax

#     def udg_recognition(self, initial_epsilon=0.7):
#         self.start_time = time.time()
#         self.last_verbose_time = self.start_time
#         self.eps = initial_epsilon
#         self.apply_granularity()
#         while True:
#             if self.verbose:
#                 print(f"Checking epsilon: {self.eps}")
#             result = self.has_discrete_realization()
#             if result == YES:
#                 self.stop_time = time.time()
#                 return True
#             if result == NO:
#                 self.stop_time = time.time()
#                 return False
#             if self.eps <= self.eps_min:
#                 self.stop_time = time.time()
#                 if self.verbose:
#                     print("Reached minimum epsilon, no realization found.")
#                 return False
            
#             self.refine_granularity()

#     def has_discrete_realization(self):
#         psi_coords_array = [None]*self.n
#         pi_permutation_order = self.bfs_order()
#         result = self.place_next_vertex(pi_permutation_order, 0, psi_coords_array)
#         return result
    
#     def place_next_vertex(self, pi_permutation_order, j: int, psi_coords_array):
#         v = pi_permutation_order[j]
#         P = self.candidate_points(pi_permutation_order, j, psi_coords_array)
#         if j == 0:
#             P = [(0.0, 0.0)]
#         if j == 1:
#             P = [p for p in P if abs(p[1]) < 1e-9 and p[1] > -1e-9]
#             if not P:
#                 P = [(self.eps, 0)]
#         if j == 2:
#             P = [p for p in P if p[0] >= -1e-9]
#             if not P:
#                 P = [(0.0, self.eps)]

#         found_trigraph = False
#         for p in P:
#             psi_coords_array[v] = p
#             if self.verbose:
#                 if time.time() - self.last_verbose_time > 10:
#                     self.last_verbose_time = time.time()
#                     print("  placing " + str(psi_coords_array))
#             if j < self.n - 1:
#                 result = self.place_next_vertex(pi_permutation_order, j+1, psi_coords_array)
#                 if result == YES:
#                     return YES
#                 if result == TRIGRAPH_ONLY:
#                     found_trigraph=True
#             else:
#                 found_trigraph = True
#                 if self.is_udg_realization(psi_coords_array):
#                     return YES
        
#         if not found_trigraph:
#             return NO
        
#         return TRIGRAPH_ONLY
    
#     def refine_granularity(self):
#         self.eps = max(self.eps / 2, self.eps_min)
#         self.apply_granularity()

#     def apply_granularity(self):
#         self.eps_sqrt2 = self.eps * SQRT_2
#         self.r_in      = 1 - self.eps_sqrt2
#         self.r_out     = 1 + self.eps_sqrt2
#         self.r_in_p2   = 1 - 2 * self.eps_sqrt2 + 2 * self.eps**2
#         self.r_out_p2  = 1 + 2 * self.eps_sqrt2 + 2 * self.eps**2

#     def candidate_points(self, pi_permutation_order, j: int, psi_coords_array):
#         v = pi_permutation_order[j]
#         neighs    = [pi_permutation_order[k] for k in range(j) if pi_permutation_order[k]     in self.neighbors(v)]
#         nonneighs = [pi_permutation_order[k] for k in range(j) if pi_permutation_order[k] not in self.neighbors(v)]

#         if not neighs:
#             x_min = -self.r_out
#             x_max =  self.r_out
#             y_min = -self.r_out
#             y_max =  self.r_out
#         else:
#             x_min = max(psi_coords_array[n][0] - self.r_out for n in neighs)
#             x_max = min(psi_coords_array[n][0] + self.r_out for n in neighs)
#             y_min = max(psi_coords_array[n][1] - self.r_out for n in neighs)
#             y_max = min(psi_coords_array[n][1] + self.r_out for n in neighs)
        
#         if x_min > x_max or y_min > y_max:
#             return []
        
#         points = []
#         start_x = int(round(x_min/self.eps))
#         end_x   = int(round(x_max/self.eps))
#         start_y = int(round(y_min/self.eps))
#         end_y   = int(round(y_max/self.eps))
#         for ix in range(start_x, end_x+1):
#             for iy in range(start_y, end_y+1):
#                 p = (ix*self.eps, iy*self.eps)
#                 good = True
#                 for n in neighs:
#                     if self.dist_gte_r_out(p, psi_coords_array[n]):
#                         good=False
#                         break
#                 if not good:
#                     continue
#                 for n in nonneighs:
#                     if self.dist_lte_r_in(p, psi_coords_array[n]):
#                         good=False
#                         break
#                 if good:
#                     points.append(p)
#         return points

#     def bfs_order(self):
#         visited = [False] * self.n
#         order = []
#         q = deque([0])
#         visited[0] = True
#         while q:
#             v = q.popleft()
#             order.append(v)
#             for w in self.neighbors(v):
#                 if not visited[w]:
#                     visited[w] = True
#                     q.append(w)
#         return order

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

def test_coordinates(verbose=False, draw=False, draw_disks=False):
    """Test graph with coordinates."""
    '''Number of vertices (<enter> to exit): 7
    V(G) = {1, 2, 3, 4, 5, 6, 7}
    Random instances (y|n)? n
    Please type the edges in the format <vertex1>,<vertex2> (e.g., 2,3)
    Next edge (<enter> if no more edges): 1,2
    Next edge (<enter> if no more edges): 1,4
    Next edge (<enter> if no more edges): 2,3
    Next edge (<enter> if no more edges): 2,5
    Next edge (<enter> if no more edges): 3,4
    Next edge (<enter> if no more edges): 3,6
    Next edge (<enter> if no more edges): 4,7
    Next edge (<enter> if no more edges): 5,6
    Next edge (<enter> if no more edges): 6,7
    Next edge (<enter> if no more edges):
    Initial granularity (type k for eps=0.7/k, recommended 1):
    Max levels deep (type 1 for no granularity refinement at all):
    Display progress (y|n)? n

    mandatory_radius = 30000
    gray_area_thickness = 0

    Final granularity = 7/40

    Coordinates (upscaled by 30000):
    1: (0,0)
    2: (28284,0)
    3: (28284,14142)
    4: (7071,28284)
    5: (56568,0)
    6: (49497,28284)
    7: (28284,49497)'''

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
    g.show_all_distances()
    if draw:
        g.draw(draw_disks)
        import matplotlib.pyplot as plt
        plt.show()

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

    if args.tests:
        tests(args.verbose)
        return
    elif args.coordinates:
        test_coordinates(args.verbose, draw=args.draw, draw_disks=args.circle)
        return
    elif args.graph6:
        g = Graph(Graph6Converter.g6_to_graph(args.graph))
    elif args.edge_list:
        g = Graph(Graph6Converter.edge_list_to_graph(args.graph))

    g.set_verbose(args.verbose)
    if args.draw:
        g.draw(args.circle)
        import matplotlib.pyplot as plt
        plt.show()
        return

    output = g.udg_recognition()
    print("Graph is " + ("" if output else "NOT ") + "a Unit Disk Graph (UDG).")


if __name__ == "__main__":
    main()

