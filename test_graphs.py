"""Test graph definitions for UDG recognition algorithm."""

from math import sin, cos, pi
from udg import Graph, LOG_BASIC

def tests(log_level=LOG_BASIC):
    # Example usage
    print("Test C3")
    g = Graph(3)
    g.set_log_level(log_level)
    g.add_edge(0,1)
    g.add_edge(1,2)
    g.add_edge(2,0)
    print("Graph C3 is UDG:", g.udg_recognition())

    print("Test P4")
    g = Graph(4)
    g.set_log_level(log_level)
    g.add_edge(0,1)
    g.add_edge(1,2)
    g.add_edge(2,3)
    print("Graph P4 is UDG:", g.udg_recognition())
    
    print("Test G5")
    g = Graph(7)
    g.set_log_level(log_level)
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
    g.set_log_level(log_level)
    g.add_edge(0,1)
    g.add_edge(0,2)
    g.add_edge(0,3)
    g.add_edge(1,4)
    g.add_edge(2,4)
    g.add_edge(3,4)
    print("Graph K2,3 is non UDG:", g.udg_recognition())

def test_coordinates_g4(log_level=LOG_BASIC):
    g = Graph(7)
    g.set_log_level(log_level)
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

def test_coordinates_g4a(log_level=LOG_BASIC):
    a = 30
    b = 5
    sin_ap = sin((a+b) * pi / 180)
    cos_ap = cos((a+b) * pi / 180)
    sin_am = sin((a-b) * pi / 180)
    cos_am = cos((a-b) * pi / 180)
    u = 30000
    e = 1.05
    g = Graph(7)
    g.set_log_level(log_level)
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

def test_coordinates_g5(log_level=LOG_BASIC):
    g = Graph(7)
    g.set_log_level(log_level)
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

def test_coordinates_g5a(log_level=LOG_BASIC):
    a = 30
    sin_a = sin(a * pi / 180)
    cos_a = cos(a * pi / 180)
    u = 30000
    e = 0.578
    f = 1.154
    g = Graph(7)
    g.set_log_level(log_level)
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

def test_coordinates_g8(log_level=LOG_BASIC):
    g = Graph(8)
    g.set_log_level(log_level)
    g.set_coordinate(0, 14142,   7071)
    g.set_coordinate(1, 49497,      0)
    g.set_coordinate(2, 14142, -28284)
    g.set_coordinate(3,     0,      0)
    g.set_coordinate(4, 28284,  21213)
    g.set_coordinate(5, 28284, -21213)
    g.set_coordinate(6, 35355, -14142)
    g.set_coordinate(7,  7071,      0)
    g.add_edge(0,3)
    g.add_edge(0,4)
    g.add_edge(0,6)
    g.add_edge(0,7)
    g.add_edge(1,4)
    g.add_edge(1,5)
    g.add_edge(1,6)
    g.add_edge(2,5)
    g.add_edge(2,6)
    g.add_edge(2,7)
    g.add_edge(3,7)
    g.add_edge(4,7)
    g.add_edge(5,6)
    g.add_edge(5,7)
    g.set_unit(30000)
    return g

def get_test_graph_by_name(graph_name: str, log_level=LOG_BASIC):
    """Get a test graph by name."""
    if graph_name == 'g4':
        return test_coordinates_g4(log_level)
    elif graph_name == 'g4a':
        return test_coordinates_g4a(log_level)
    elif graph_name == 'g5':
        return test_coordinates_g5(log_level)
    elif graph_name == 'g5a':
        return test_coordinates_g5a(log_level)
    elif graph_name == 'g8':
        return test_coordinates_g8(log_level)
    else:
        raise ValueError(f"Unknown test graph name: {graph_name}")
