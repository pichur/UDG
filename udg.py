"""Simple Unit Disk Graph recognition algorithm."""

from math import sqrt
from collections import deque
import networkx as nx

class Graph:
    """Simple adjacency list graph that can be built from an integer number
    of vertices or from a :class:`networkx.Graph` instance."""

    def __init__(self, n_or_g):
        if isinstance(n_or_g, int):
            n = n_or_g
            self.n = n
            self.adj = [[] for _ in range(n)]
        elif isinstance(n_or_g, nx.Graph):
            mapping = {node: idx for idx, node in enumerate(sorted(n_or_g.nodes()))}
            self.n = len(mapping)
            self.adj = [[] for _ in range(self.n)]
            for u, v in n_or_g.edges():
                self.add_edge(mapping[u], mapping[v])
        else:
            raise TypeError("Graph() expects an int or a networkx.Graph")

    def add_edge(self, u, v):
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)

    def neighbors(self, v):
        return self.adj[v]

    def is_edge(self, u, v):
        return v in self.adj[u]

def bfs_order(graph: Graph):
    visited = [False]*graph.n
    order = []
    q = deque([0])
    visited[0]=True
    while q:
        v=q.popleft()
        order.append(v)
        for w in graph.neighbors(v):
            if not visited[w]:
                visited[w]=True
                q.append(w)
    return order


def dist(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def refine_granularity(eps: float) -> float:
    return eps/2


TRIGRAPH_ONLY = "TRIGRAPH_ONLY"
YES = "YES"
NO = "NO"


def is_udg_realization(graph: Graph, eps: float, coords):
    n = graph.n
    for i in range(n):
        for j in range(i+1, n):
            d = dist(coords[i], coords[j])
            if graph.is_edge(i,j):
                if d > 1 + 1e-6:
                    return False
            else:
                if d < 1 - 1e-6:
                    return False
    return True


def candidate_points(graph: Graph, eps: float, pi, j, coords):
    v = pi[j]
    neighs = [pi[k] for k in range(j) if pi[k] in graph.neighbors(v)]
    nonneighs = [pi[k] for k in range(j) if pi[k] not in graph.neighbors(v)]
    r_in = 1 + eps*sqrt(2)
    r_out = 1 - eps*sqrt(2)

    if not neighs:
        x_min = -r_in
        x_max = r_in
        y_min = -r_in
        y_max = r_in
    else:
        x_min = max(coords[n][0] - r_in for n in neighs)
        x_max = min(coords[n][0] + r_in for n in neighs)
        y_min = max(coords[n][1] - r_in for n in neighs)
        y_max = min(coords[n][1] + r_in for n in neighs)
    if x_min > x_max or y_min > y_max:
        return []
    points = []
    start_x = int(round(x_min/eps))
    end_x = int(round(x_max/eps))
    start_y = int(round(y_min/eps))
    end_y = int(round(y_max/eps))
    for ix in range(start_x, end_x+1):
        for iy in range(start_y, end_y+1):
            p = (ix*eps, iy*eps)
            good = True
            for n in neighs:
                if dist(p, coords[n]) >= r_in:
                    good=False
                    break
            if not good:
                continue
            for n in nonneighs:
                if dist(p, coords[n]) <= r_out:
                    good=False
                    break
            if good:
                points.append(p)
    return points


def place_next_vertex(graph: Graph, eps: float, pi, j: int, coords):
    v = pi[j]
    P = candidate_points(graph, eps, pi, j, coords)
    if j == 0:
        P = [(0.0, 0.0)]
    if j == 1:
        P = [p for p in P if abs(p[1]) < 1e-9 and p[0] > -1e-9]
        if not P:
            P = [(eps, 0.0)]
    if j == 2:
        P = [p for p in P if p[1] >= -1e-9]
        if not P:
            P = [(0.0, eps)]
    found_trigraph = False
    for p in P:
        coords[v] = p
        if j == graph.n -1:
            found_trigraph = True
            if is_udg_realization(graph, eps, coords):
                return YES
        else:
            result = place_next_vertex(graph, eps, pi, j+1, coords)
            if result == YES:
                return YES
            if result == TRIGRAPH_ONLY:
                found_trigraph=True
    if not found_trigraph:
        return NO
    return TRIGRAPH_ONLY


def has_discrete_realization(graph: Graph, eps: float):
    coords=[None]*graph.n
    pi=bfs_order(graph)
    result=place_next_vertex(graph, eps, pi, 0, coords)
    return result


def udg_recognition(graph: Graph, initial_epsilon=0.7, eps_min=1e-3, verbose=False):
    eps = initial_epsilon
    while True:
        if verbose:
            print(f"Checking epsilon: {eps}")
        result = has_discrete_realization(graph, eps)
        if result == YES:
            return True
        if result == NO or eps <= eps_min:
            return False
        eps = max(refine_granularity(eps), eps_min)

if __name__ == "__main__":
    # Example usage
    g = Graph(3)
    g.add_edge(0,1)
    g.add_edge(1,2)
    g.add_edge(0,2)
    print("Graph is UDG:", udg_recognition(g))
