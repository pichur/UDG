# ---------------------------------------------- 
#                                                       
#    UNIT DISC GRAPH RECOGNITION                              
#                                                           
# ----------------------------------------------

from asyncio import graph
import math, sys, time, datetime
from random import random, uniform, randint

import argparse
import Graph6Converter
import networkx as nx

## log levels
NONE  = 0 # only results
INFO  = 1 # include info level msgs 
DEBUG = 2 # include info and debug level msgs
HIGH  = 3 # highest debug verbosity
LOG_LEVEL = NONE ## select intended log level

## progress
PROGRESS_STEP = 1000 # number of attempted configurations
START_TIME = 0

## grid parameters
HALF_OF_GRAY_AREA_THICKNESS = 10000 # x (half of the optional neighborhood circular crown thickness)
GRID_SPACEMENT = 7071 # g < x/sqrt(2)

## max computation time per instance (in seconds)
ONE_WEEK = 60 * 60 * 24 * 7
MAX_TIME_YES_OPTIMIZATION = ONE_WEEK
MAX_TIME_GLOBAL = ONE_WEEK

## input options
SIMPLIFIED_INPUT = True
MULTIPLE_RANDOM_INSTANCES = False
RANDOM_UDG_MODEL = False
HARDCODED_INSTANCES = False


######################################

# Global variables for tracking work
collect_work_summary: bool = False
place_next_vertex_counter: int = 0
place_next_vertex_by_level_and_order_counter: list[list[int]] = []

######################################

# progress?
DISPLAY_PROGRESS = False

## placement options
SECOND_VERTEX_ZERO_ORDINATE = True
THIRD_VERTEX_NON_NEGATIVE_ORDINATE = True
FIRST_PLACED_SYMMETRIC_NON_NEGATIVE_COORDINATE = True
APPLY_FILTER_TO_LAST_VERTEX = False

## file?
FROM_FILE = False # defaults to standard stdin

## optimization
OPTIMIZE_FOR_YES = False
OPTIMIZE_FOR_NO = False
TIMEOUT = -1
NEXT_TIME_LIMIT = None

## recursion
MIN_K = 1
MAX_RECURSIVE_LEVELS = 3

## results
YES = 1
NO = 0
INCONCLUSIVE = -1

## random instances 
RANDOM = False
EDGE_PROB = None
#PACKING_FACTOR = None
N_RANDOM_INSTANCES = None

## edge types
MANDATORY_EDGE = 1
FORBIDDEN_EDGE = 2
OPTIONAL_EDGE_FOR_MANDATORY = 3
OPTIONAL_EDGE_FOR_FORBIDDEN = 4

## fixed-permutation groups
CENTER = 0
SATELLITES = 1

## graph attributes
NUMBER_OF_VERTICES = 0
NEIGHBORS = 1
PLACEMENT_ORDER = 2
PREDESCESSORS = 3
FIXED_SECTORS = 4
SYMMETRIC_VERTICES = 5
COORDINATES = 6

GRAPH_ATTRIBUTES_COUNT = 7 # must keep up-to-date

#    graph[0] : an integer giving the number of vertices
#    graph[1] : a dictionary, whose keys are integers from 1 to n representing
#               each vertex v, and whose values are sets of neighbors of v
#    graph[2] : the order in which the vertices will be placed in the grid
#    graph[3] : a list of vertex predescessors (according to the ordered groups input by the user)
#    graph[4] : a dictionary, mapping each vertex to its fixed circular sector (if any)
#    graph[5] : a dictionary, mapping each vertex to its mirror (if any)
#    graph[6] : coordinates of a possible UDG sandwich model for the graph (if any)

# ---------------------------------------------------------


def spillguts(x):
    if (type(x) == frozenset) or (type(x) == set):
        result = "{"
        for element in x:
            result += spillguts(element) + ","
        if len(x) > 0:
            result = result[:-1]
        result += "}"
        return result
    elif type(x) == list:
        result = "["
        for element in x:
            result += spillguts(element) + ","
        if len(x) > 0:
            result = result[:-1]
        result += "]"
        return result  
    elif type(x) == tuple:
        result = "("
        for element in x:
            result += spillguts(element) + ","
        if len(x) > 0:
            result = result[:-1]
        result += ")"
        return result  
    else:
        return str(x)
    
def info(msg):
    if LOG_LEVEL >= INFO:
        print(msg)
        
def debug(msg, level = DEBUG):
    if LOG_LEVEL >= level:
        print(msg)

# ---------------------------------------------------------

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def obtain_standard_deviation(elements):
    std_dev = 0
    if len(elements) > 0:
        avg = sum(elements)/len(elements)
        for element in elements:
            dev = element - avg
            std_dev += dev**2
        std_dev /= len(elements)
        std_dev = std_dev**0.5
    return std_dev

def select_uniform_point_circle(r):
    t = 2 * math.pi * random()
    d = random() + random()
    if d > 1:
        d = 2 - d
    return (r*d*math.cos(t), r*d*math.sin(t))

def is_angle_less_than_pi(A, O, B):
    # positive if A-->O-->B is a right turn
    return ((A[0]-O[0])*(B[1]-O[1]) - (A[1]-O[1])*(B[0]-O[0]) >= 0) 

def angle(A, O, B):
    if (A==O) or (B==O):
        return 0
    if (-A[0],-A[1]) == B:
        return math.pi
    v1 = (A[0]-O[0],A[1]-O[1])
    v2 = (B[0]-O[0],B[1]-O[1])
    try:
        result = math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
    except ValueError:
        return angle(A, (O[0]+1,O[1]), B) # workaround for precision issues
    if not is_angle_less_than_pi(A, O, B):
        result = 2 * math.pi - result
    return result

def square_distance(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def get_max_square_distance_between_cell_points(bottomleft1, bottomleft2):
    x1, y1 = bottomleft1[0], bottomleft1[1]
    x2, y2 = bottomleft2[0], bottomleft2[1]

    if x1 <= x2:
        if y1 <= y2:
            return square_distance((x1, y1), (x2 + GRID_SPACEMENT, y2 + GRID_SPACEMENT))
        else:
            return square_distance((x1, y1 + GRID_SPACEMENT), (x2 + GRID_SPACEMENT, y2))
    else:
        if y1 <= y2:
            return square_distance((x1 + GRID_SPACEMENT, y1), (x2, y2 + GRID_SPACEMENT))
        else:
            return square_distance((x1 + GRID_SPACEMENT, y1 + GRID_SPACEMENT), (x2, y2))

def get_min_square_distance_between_cell_points(bottomleft1, bottomleft2):
    x1, y1 = bottomleft1[0], bottomleft1[1]
    x2, y2 = bottomleft2[0], bottomleft2[1]

    if x1 < x2:
        if y1 < y2:
            return square_distance((x1 + GRID_SPACEMENT, y1 + GRID_SPACEMENT), (x2, y2))
        elif y1 == y2:
            return square_distance((x1 + GRID_SPACEMENT, y1), (x2, y2))
        else:
            return square_distance((x1 + GRID_SPACEMENT, y1), (x2, y2 + GRID_SPACEMENT))
    elif x1 == x2:
        if y1 < y2:
            return square_distance((x1, y1 + GRID_SPACEMENT), (x2, y2))
        elif y1 == y2:
            return 0
        else:
            return square_distance((x1, y1), (x2, y2 + GRID_SPACEMENT))
    else:
        if y1 < y2:
            return square_distance((x1, y1 + GRID_SPACEMENT), (x2 + GRID_SPACEMENT, y2))
        elif y1 == y2:
            return square_distance((x1, y1), (x2 + GRID_SPACEMENT, y2))
        else:
            return square_distance((x1, y1), (x2 + GRID_SPACEMENT, y2 + GRID_SPACEMENT))

def get_neighborhood_points(label, point, radius, include_gray_zone):
    result = set()

    if include_gray_zone:
        allowed_distance = radius + HALF_OF_GRAY_AREA_THICKNESS
    else:
        allowed_distance = radius - HALF_OF_GRAY_AREA_THICKNESS

    allowed_square_distance = allowed_distance**2

    # bounding square
    trim = allowed_distance % GRID_SPACEMENT
    left = point[0] - allowed_distance + trim
    right = point[0] + allowed_distance - trim
    bottom = point[1] - allowed_distance + trim
    top = point[1] + allowed_distance - trim

    # actual grid points within neighboring distance
    for x in range(left, right+1, GRID_SPACEMENT):
        for y in range(bottom, top+1, GRID_SPACEMENT):

            # old way 
            if (((include_gray_zone) and (square_distance((x,y),point) <= allowed_square_distance)) or
                ((not include_gray_zone) and (square_distance((x,y),point) < allowed_square_distance))):
                result.add((x,y))

## ENHANCEMENT: improve the allowed distances based on actual relative position
##
##            if (include_gray_zone and (get_max_square_distance_between_cell_points(point, (x,y)) <= (2 * radius)**2) or \
##                not include_gray_zone and (get_min_square_distance_between_cell_points(point, (x,y)) < (2 * radius)**2)):
##                result.add((x,y))
     
    return result 

# ---------------------------------------------------------
        
def has_edge(graph, edge):
    return edge[1] in graph[NEIGHBORS][edge[0]]

def add_edge(graph, edge):
    graph[NEIGHBORS][edge[0]].add(edge[1])
    graph[NEIGHBORS][edge[1]].add(edge[0])

def clear_edges(graph):
    graph[NEIGHBORS] = [None]
    for i in range(graph[NUMBER_OF_VERTICES]):
        graph[NEIGHBORS] += [set()]

def get_degree(graph, vertex):
    return len(graph[NEIGHBORS][vertex])

def clear_coordinates(graph):
    n = graph[NUMBER_OF_VERTICES]
    graph[COORDINATES] = [0] + [None]*n

def create_empty_graph(n:int) -> list:
    G = [None] * GRAPH_ATTRIBUTES_COUNT

    G[NUMBER_OF_VERTICES] = n

    # 1-based indexing for adjacency sets to match vertex ids
    clear_edges(G)

    G[PLACEMENT_ORDER] = None
    G[PREDESCESSORS] = {}
    G[FIXED_SECTORS] = {}
    G[SYMMETRIC_VERTICES] = {}

    clear_coordinates(G)

    return G

def get_breadth_first_search_order(G, root):
    result = []

    n = G[NUMBER_OF_VERTICES]

    visited = [False]*(n+1)
    visited[root] = True
    result += [root]  
    queue = [root]

    while len(queue) > 0:
        v = queue[0]
        neighbors = G[NEIGHBORS][v]
        for w in neighbors:
            if not visited[w]:
                visited[w] = True
                queue += [w]               
                result += [w]    
        del(queue[0])

    return result

def compare_points(pointA, pointB):
    if pointA[0] != pointB[0]:
        return pointA[0] - pointB[0]
    return pointA[1] - pointB[1]

def position_conforms_to_ordered_groups(vertex, candidate_position, predescessors, coordinates):
    predescessor = predescessors.get(vertex, None)
    if predescessor == None:
        return True
    return compare_points(candidate_position, coordinates[predescessor]) >= 0

def is_UDG_model(coordinates, graph, radius):
    n = graph[NUMBER_OF_VERTICES]

    has_optional_edge_for_mandatory = False
    has_optional_edge_for_forbidden = False

    gray_zone_min_square_distance = (radius - HALF_OF_GRAY_AREA_THICKNESS)**2
    gray_zone_max_square_distance = (radius + HALF_OF_GRAY_AREA_THICKNESS)**2

    for a in range(1, n):
        for b in range (a+1, n+1):
            d2 = square_distance(coordinates[a],coordinates[b]) 
            if ((d2 > gray_zone_min_square_distance) and
                (d2 <= gray_zone_max_square_distance)):
                if has_edge(graph, (a, b)):
                    has_optional_edge_for_mandatory = True
                else:
                    has_optional_edge_for_forbidden = True

    return (not has_optional_edge_for_mandatory) or (not has_optional_edge_for_forbidden)

def describe_sandwich_model_edges(graph, radius):
    adjacencies_summary = {}
    square_distances = {}

    n = graph[NUMBER_OF_VERTICES]
    coordinates = graph[COORDINATES]

    for a in range(1, n):
        for b in range (a+1, n+1):
            dist2 = square_distance(coordinates[a],coordinates[b])
            if dist2 <= (radius - HALF_OF_GRAY_AREA_THICKNESS)**2:
                edge_type = MANDATORY_EDGE
            elif dist2 > (radius + HALF_OF_GRAY_AREA_THICKNESS)**2:
                edge_type = FORBIDDEN_EDGE
            else:    
                if has_edge(graph, (a,b)):
                    edge_type = OPTIONAL_EDGE_FOR_MANDATORY
                else:
                    edge_type = OPTIONAL_EDGE_FOR_FORBIDDEN
                
            adjacencies_summary[(a,b)] = edge_type
            square_distances[(a,b)] = dist2

    return adjacencies_summary, square_distances

# ---------------------------------------------------------

def read_number_of_vertices():
    try:
        result = eval(input("\nNumber of vertices (<enter> to exit): "))
    except:
        result = 0
    if FROM_FILE:
        print(result)
    if result > 0:
        print("V(G) = {", end="")
        for i in range(1, result):
            print("%d, " % i, end="")
        print("%d}" % result)
    return result

def set_placement_order(graph, custom_order:str = "bfs"):
    n = graph[NUMBER_OF_VERTICES]

    result = None

    if (custom_order.startswith("bfs")):
        # BFS with optional root specification
        if ":" in custom_order:
            parts = custom_order.split(":", 1)
            bfs_root = int(parts[1])
        else:
            min_degree = n
            min_degree_vertex = None
            for v in range(1, n+1):
                degree = get_degree(graph, v)
                if degree < min_degree:
                    min_degree = degree
                    min_degree_vertex = v

            bfs_root = min_degree_vertex
        
        result = get_breadth_first_search_order(graph, bfs_root)

    elif custom_order == "c":
        # crescent order
        order_list = []
        for i in range(1, n+1):
            order_list += [i]
        order_tuple = tuple(order_list)            

    else:
        # user-defined order
        order_tuple = eval(custom_order)
        if len(order_tuple) != n:
            print("Invalid order. Placement orders must have %d vertices." % n)
            exit()

    if result == None:
        result = [order_tuple[0]] # the first vertex in the custom order has no parent

        for i in range(1, len(order_tuple)):
            vertex = order_tuple[i]
            parent = None
            for j in range(i-1, -1, -1):
                parent_candidate = order_tuple[j]
                if has_edge(graph, (vertex, parent_candidate)):
                    parent = parent_candidate
                    break
            if parent == None:
                print("Invalid order. Vertices must be placed in a connected fashion.")
                exit()
            
            result += [vertex]
    
    info("Placement order = " + spillguts(result))

    graph[PLACEMENT_ORDER] = result

def read_ordered_groups(graph):
    predescessors = {}
    placement_order = graph[PLACEMENT_ORDER]
    
    print("Please enter the ordered groups (center first, comma-separated)")

    while True:
        group_string = input("Next group: ")
        if FROM_FILE:
            print(group_string)
            
        if group_string == "":
            break

        group_list = list(eval(group_string))

        if len(group_list) < 2:
            print("Each group must comprise at least 2 vertices.")
            continue

        ordered_group_list = []
        for v in placement_order:
            if v in group_list:
                ordered_group_list += [v]

        error = False
        for index in range(1, len(ordered_group_list)):
            vertex = ordered_group_list[index]
            predescessor = ordered_group_list[index - 1]
            if predescessors.get(vertex, None) != None:
                error = True
                break
            predescessors[vertex] = predescessor

        if error:
            print("Each vertex can appear only once in the ordered groups. Please reenter all ordered groups.")
            predescessors.clear()
            continue
                
    graph[PREDESCESSORS] = predescessors
    debug("predescessors = " + spillguts(predescessors))

def read_fixed_sectors(graph):
    result = {}

    print("Enter the vertices with fixed sectors (<vertex>,<start_angle>,<end_angle>)")

    while True:
        fixed_sector_string = input("Next fixed sector: ")
        if FROM_FILE:
            print(fixed_sector_string)
            
        if fixed_sector_string == "":
            break

        fixed_sector = list(eval(fixed_sector_string))
        vertex = fixed_sector[0]
        start_angle = math.radians(fixed_sector[1])
        end_angle = math.radians(fixed_sector[2])
        result[vertex] = (start_angle, end_angle)

    graph[FIXED_SECTORS] = result
    debug("fixed_sectors = " + spillguts(result))

def read_symmetric_vertices(graph):
    result = {}

    print("Enter the symmetric vertices (<vertex1>,<vertex2>,h|v)")

    while True:
        symmetric_vertices_string = input("Next pair of symmetric vertices: ")
        if FROM_FILE:
            print(symmetric_vertices_string)
            
        if symmetric_vertices_string == "":
            break

        symmetric_vertices = symmetric_vertices_string.split(",")
        vertex1 = int(symmetric_vertices[0])
        vertex2 = int(symmetric_vertices[1])
        symmetry_type = symmetric_vertices[2]
        result[vertex1] = (vertex2, symmetry_type)

    graph[SYMMETRIC_VERTICES] = result
    debug("symmetric_vertices = " + spillguts(result))

def read_placement_options(graph):
    global SECOND_VERTEX_ZERO_ORDINATE
    global THIRD_VERTEX_NON_NEGATIVE_ORDINATE
    global APPLY_FILTER_TO_LAST_VERTEX
    global FIRST_PLACED_SYMMETRIC_NON_NEGATIVE_COORDINATE

    if graph[NUMBER_OF_VERTICES] >= 2:
        _2nd_vertex_option = input("Force 2nd vertex with ordinate zero (y|n)? ")
        SECOND_VERTEX_ZERO_ORDINATE = (_2nd_vertex_option in "Yy")
        if FROM_FILE:
            print(SECOND_VERTEX_ZERO_ORDINATE)
        if SECOND_VERTEX_ZERO_ORDINATE and \
           ((graph[PLACEMENT_ORDER][1] in graph[PREDESCESSORS].keys()) or \
            (graph[PLACEMENT_ORDER][1] in graph[PREDESCESSORS].values())):
            print("Cannot restrict 2nd vertex position because it belongs to an ordered group.")
            SECOND_VERTEX_ZERO_ORDINATE = False

    if graph[NUMBER_OF_VERTICES] >= 3:
        _3rd_vertex_option = input("Force 3rd vertex with non-negative ordinate (y|n)? ")
        THIRD_VERTEX_NON_NEGATIVE_ORDINATE = (_3rd_vertex_option in "Yy")
        if FROM_FILE:
            print(THIRD_VERTEX_NON_NEGATIVE_ORDINATE)
        if THIRD_VERTEX_NON_NEGATIVE_ORDINATE and \
           ((graph[PLACEMENT_ORDER][2] in graph[PREDESCESSORS].keys()) or \
            (graph[PLACEMENT_ORDER][2] in graph[PREDESCESSORS].values())):
            print("Cannot restrict 3rd vertex position because it belongs to an ordered group.")
            THIRD_VERTEX_NON_NEGATIVE_ORDINATE = False
        
    first_placed_symmetric_non_negative_coordinate = input("Force non-negative abscissa/ordinate "\
                                        "to first placed symmetric vertex (y|n)? ")
    FIRST_PLACED_SYMMETRIC_NON_NEGATIVE_COORDINATE = (first_placed_symmetric_non_negative_coordinate not in "Nn")                  
    if FROM_FILE:
        print(FIRST_PLACED_SYMMETRIC_NON_NEGATIVE_COORDINATE)

    apply_filter_to_last_vertex = input("Apply filter to last vertex (y|n)? ")
    APPLY_FILTER_TO_LAST_VERTEX = (apply_filter_to_last_vertex not in "Nn")
    if FROM_FILE:
        print(APPLY_FILTER_TO_LAST_VERTEX)

def get_elapsed_time():
    result = time.time() - START_TIME
    return result

def format_time(duration, full_format = False):
    result = ""
    if duration < 10:
        result += "%d ms" % (duration * 1000)
    else:
        days, hours, minutes, seconds = 0, 0, 0, 0
        if duration > 24*60*60:
            days = int(duration/(24*60*60))
            duration = duration % (24*60*60)
        if duration > 60*60:
            hours = int(duration/(60*60))
            duration = duration % (60*60)
        if duration > 60:
            minutes = int(duration/60)
            duration = duration % 60
        seconds = duration

        if days > 0 or not full_format:
            plural = ""
            if days > 1:
                plural = "s"
            if full_format:
                result += "%d day" % days + plural + ", "
            else:
                if days >= 1:
                    result += "%d" % days + " day" + plural + " + "

        if full_format or (days < 10):
                                
            if hours > 0 or not full_format:
                plural = ""
                if hours > 1:
                    plural = "s"
                if full_format:
                    result += "%d hour" % hours + plural + ", "
                else:
                    if hours < 10:
                        result += "0"
                    result += "%d" % hours + ":"
            if minutes > 0 or not full_format:
                plural = ""
                if minutes > 1:
                    plural = "s"
                if full_format:
                    result += "%d minute" % minutes + plural + ", "
                else:
                    if minutes < 10:
                        result += "0"
                    result += "%d" % minutes + ":"
            if seconds > 0 or not full_format:
                plural = ""
                if seconds > 1:
                    plural = "s"
                if full_format:
                    result += "%d second" % seconds + plural + ", "
                else:
                    if seconds < 10:
                        result += "0"
                    result += "%d" % seconds + "  "

        result = result[:-2]

    return result

def obtain_printable_model(graph, convert_optional_edges):
    result = ""

    latest_parameter_k = graph[COORDINATES][0]       
    radius = latest_parameter_k * HALF_OF_GRAY_AREA_THICKNESS

    adjacencies_summary, square_distances = describe_sandwich_model_edges(graph, radius)

    result += "\nFinal granularity = 7/%d" % (10*latest_parameter_k)

    mandatory_radius = radius - HALF_OF_GRAY_AREA_THICKNESS
    gray_area_thickness = 2 * HALF_OF_GRAY_AREA_THICKNESS

    if convert_optional_edges:
        if OPTIONAL_EDGE_FOR_MANDATORY in adjacencies_summary.values():
            mandatory_radius += gray_area_thickness
        gray_area_thickness = 0
        scale = mandatory_radius
    else:
        scale = radius

    print("mandatory_radius =", mandatory_radius)
    print("gray_area_thickness =", gray_area_thickness)

    result += "\n\nCoordinates (upscaled by %d):" % scale

        
    #result += "\nMandatory adjacency distance = %d" % mandatory_radius
    #result += "\nGray area thickness = %d" % gray_area_thickness   
    
    n = graph[NUMBER_OF_VERTICES]
    for i in range(1, n+1):
        coordinate = graph[COORDINATES][i]
        result += "\n" + str(i) + ": " + spillguts(coordinate)

    result += "\n\nAdjacency types and distances (upscaled by %d):" % scale
    for pair_of_vertices in sorted(adjacencies_summary.keys()):
        edge_type = adjacencies_summary[pair_of_vertices]
        dist2 = square_distances[pair_of_vertices]
        if edge_type == MANDATORY_EDGE: 
            adjacency_string = "mandatory"
        elif edge_type == FORBIDDEN_EDGE:
            adjacency_string = "forbidden"
        elif edge_type == OPTIONAL_EDGE_FOR_MANDATORY:
            if convert_optional_edges:
                adjacency_string = "mandatory"
            else:
                adjacency_string = "optional (for neighbors in G)"
        elif edge_type == OPTIONAL_EDGE_FOR_FORBIDDEN:
            if convert_optional_edges:
                adjacency_string = "forbidden"
            else:
                adjacency_string = "optional (for non-neighbors in G)"
            
        result += "\n(" + \
                  str(pair_of_vertices[0]) + "," + \
                  str(pair_of_vertices[1]) + ") = " + \
                  adjacency_string + \
                  " [distance = %.1f]" % (math.sqrt(dist2))

    return result
    

# ---------------------------------------------------------

def get_formatted_datetime():
    now = datetime.datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")

def increment_progress_level(progress, vertex_count, recursive_level, already_placed):
    base_index = vertex_count * recursive_level + 1
    index = base_index + already_placed
    progress[index][0] += 1

def increment_progress_total_attempts(progress):
    progress[0] += 1

def reset_current_progress_level(progress, vertex_count, recursive_level,
                                 already_placed, candidate_positions_count):
    base_index = vertex_count * recursive_level + 1
    while len(progress) < base_index + 1:
        for i in range(vertex_count):
            progress += [[0,0]]
    index = base_index + already_placed
    progress[index][0] = 0
    progress[index][1] = candidate_positions_count

def display_progress(progress, vertex_count, recursive_level, already_placed):    

    ratio_sequence = ""

    level = 0
    k = MIN_K

    while level <= recursive_level:
        if OPTIMIZE_FOR_YES and level < recursive_level:
            level += 1
            k = k*2
            continue
        ratio_sequence += "  " * level
        ratio_sequence += "{at eps=7/%d} " % (10*k)
        start_index = vertex_count * level + 1
        if level < recursive_level:
            end_index = start_index + vertex_count - 1
        else:
            end_index = start_index + already_placed
        for i in range(start_index, end_index + 1):
            ratio = "[%d/%d]" % (progress[i][0], progress[i][1])
            ratio_sequence += "%s" % ratio
        if level < recursive_level:
            ratio_sequence += "\n"
        else:
            if already_placed < vertex_count - 1:
                ratio_sequence += "..."  # "(%d)..." % (already_placed + 1)
        level += 1
        k = k*2

    if OPTIMIZE_FOR_YES:
        start_index = vertex_count * recursive_level + 1
    else:
        start_index = 1
    end_index = vertex_count * recursive_level + already_placed

    elapsed = get_elapsed_time()
    
    work_completed = 0
    current_denominator = 1
    for i in range(start_index, end_index + 1):
        work_completed += (progress[i][0] - 1) / (progress[i][1] * current_denominator)
        current_denominator *= progress[i][1]
    if work_completed > 0:
        remaining = format_time(((1 - work_completed) / work_completed) * elapsed)
    else:
        remaining = "unknown"

    print("\n" + get_formatted_datetime() + "      %.8f%% done" % (work_completed * 100),
          "\nelapsed time:", format_time(elapsed),
          "  remaining time:", remaining, "(rough estimate)\nAttempted placements by vertex:\n",
          ratio_sequence)

def initialize_progress():
    return [0]

# ---------------------------------------------------------

def determine_possible_locations(graph, already_placed, coordinates, radius, model=set()): 
    n = graph[NUMBER_OF_VERTICES]
    placement_order = graph[PLACEMENT_ORDER]
    predescessors = graph[PREDESCESSORS]
    fixed_sectors = graph[FIXED_SECTORS]
    symmetric_vertices = graph[SYMMETRIC_VERTICES]

    vertex = placement_order[already_placed]

    symmetric_tuple = symmetric_vertices.get(vertex, (None, None))
    symmetric_vertex = symmetric_tuple[0]
    symmetry_type = symmetric_tuple[1]

    debug("vertex = " + spillguts(vertex))
    debug("already_placed = " + spillguts(already_placed))
    debug("coordinates = " + spillguts(coordinates))
    debug("model = " + spillguts(model))

    allowed_points = set()
    forbidden_points = set()

    # 1st vertex always at the origin (by translation)
    if already_placed == 0:
        return [(0,0)] 

    # when a sandwich model is given, the possible locations of the current vertex will be
    # a subset of 4 points: (i)   the base point itself,
    #                       (ii)  the base point shifted one grid unit to the right,
    #                       (iii) the base point shifted one grid unit up
    #                       (iv)  the base point shifted one grid unit up and one grid unit to the right
    if len(model) > 0:
        base_point = model[vertex]
        if base_point != None:
            model_compliant_points = set()
            model_compliant_points.add(base_point)
            model_compliant_points.add((base_point[0] + GRID_SPACEMENT, base_point[1]))
            model_compliant_points.add((base_point[0], base_point[1] + GRID_SPACEMENT))
            model_compliant_points.add((base_point[0] + GRID_SPACEMENT, base_point[1] + GRID_SPACEMENT))
            if len(allowed_points) == 0:
                allowed_points |= model_compliant_points
            else:
                allowed_points &= model_compliant_points
                if len(allowed_points) == 0:
                    return []
        
    # for each already placed neighbor v...
    for i in range(already_placed):
        v = placement_order[i]
        if not has_edge(graph, (v, vertex)):
            continue

        if len(allowed_points) == 0:
            allowed_points |= get_neighborhood_points(v, coordinates[v], radius, not OPTIMIZE_FOR_YES)
        else:
            # removes the points that do not belong to the mandatory+optional neighborhood area of v
            forbidden_points.clear()
            for point in allowed_points:
                if OPTIMIZE_FOR_YES:
                    neighborhood_radius = radius - HALF_OF_GRAY_AREA_THICKNESS
                else:
                    neighborhood_radius = radius + HALF_OF_GRAY_AREA_THICKNESS
                if square_distance(coordinates[v], point) >= neighborhood_radius**2:
                    forbidden_points.add(point)
            allowed_points -= forbidden_points
            if len(allowed_points) == 0:
                break

    # for each already placed non-neighbor v...
    for i in range(already_placed):
        v = placement_order[i]
        if has_edge(graph, (v, vertex)):
            continue

        # removes the points that belong to the mandatory neighborhood area of v
        forbidden_points.clear()
        for point in allowed_points:
            if square_distance(coordinates[v], point) <= (radius - HALF_OF_GRAY_AREA_THICKNESS)**2:
                forbidden_points.add(point)
        allowed_points -= forbidden_points
        if len(allowed_points) == 0:
            break
    
## ENHANCEMENT: the 'Four Neighboring Cells' Theorem
##
##    false_positive_points = set()     
##    for allowed_point in allowed_points:
##        # vertical shift
##        neighboring_point_1 = (allowed_point[0], allowed_point[1] + GRID_SPACEMENT)
##        neighboring_point_2 = (allowed_point[0], allowed_point[1] - GRID_SPACEMENT)
##        if (neighboring_point_1 not in allowed_points) and \
##           (neighboring_point_2 not in allowed_points):
##            false_positive_points.add(allowed_point)
##            continue
##
##        # horizontal shift
##        neighboring_point_1 = (allowed_point[0] + GRID_SPACEMENT, allowed_point[1])
##        neighboring_point_2 = (allowed_point[0] - GRID_SPACEMENT, allowed_point[1])
##        if (neighboring_point_1 not in allowed_points) and \
##           (neighboring_point_2 not in allowed_points):
##            false_positive_points.add(allowed_point)
##            continue
##        
##        # diagonal shift
##        neighboring_point_1 = (allowed_point[0] + GRID_SPACEMENT, allowed_point[1] + GRID_SPACEMENT)
##        neighboring_point_2 = (allowed_point[0] + GRID_SPACEMENT, allowed_point[1] - GRID_SPACEMENT)
##        neighboring_point_3 = (allowed_point[0] - GRID_SPACEMENT, allowed_point[1] + GRID_SPACEMENT)
##        neighboring_point_4 = (allowed_point[0] - GRID_SPACEMENT, allowed_point[1] - GRID_SPACEMENT)
##        if (neighboring_point_1 not in allowed_points) and \
##           (neighboring_point_2 not in allowed_points) and \
##           (neighboring_point_3 not in allowed_points) and \
##           (neighboring_point_4 not in allowed_points):
##            false_positive_points.add(allowed_point)
##        
##    allowed_points -= false_positive_points
##
##    print("\nvertex =", vertex)
##    print("allowed_points =", allowed_points)
        
    # extra input filters            
    if ((len(allowed_points) > 0) and
        ((already_placed < n-1) or APPLY_FILTER_TO_LAST_VERTEX)):

        filtered_out = set()
        for point in allowed_points:
            if SECOND_VERTEX_ZERO_ORDINATE:
                # 2nd vertex always with positive abscissa and ordinate 0 (by rotation)
                if ((already_placed == 1) and ((point[0] <= 0) or (point[1] != 0))):
                    filtered_out.add(point)
        
            if THIRD_VERTEX_NON_NEGATIVE_ORDINATE:
                # 3rd vertex always with non-negative ordinate (by reflection)
                if ((already_placed == 2) and (point[1] < 0)):
                    filtered_out.add(point)

            # the coordinate of the vertex must be >= its predescessor (if any) in the ordered group it belongs to
            if not position_conforms_to_ordered_groups(vertex, point, predescessors, coordinates):
                filtered_out.add(point)
                    
            # filters out grid points that do not fall into the intended circular sector 
            if vertex in fixed_sectors:
                start_angle = fixed_sectors[vertex][0]
                end_angle = fixed_sectors[vertex][1]
                point_angle = angle((GRID_SPACEMENT,0),(0,0),point)
                if (point_angle < start_angle) or (point_angle > end_angle):
                    filtered_out.add(point)

            if symmetric_vertex != None:
                if symmetric_vertex == vertex:
                # a vertex v whose user-defined symmetric is v itself will only be assigned
                # coordinates with abscissa (resp. ordinate) zero, for vertical (resp. horizontal) symmetry
                    if (((symmetry_type in "vV") and (point[0] != 0)) or
                        ((symmetry_type in "hH") and (point[1] != 0))):
                        filtered_out.add(point)
                        ##print("non_compliant with auto-symmetry: ", point)
                elif (coordinates[symmetric_vertex] == None) and FIRST_PLACED_SYMMETRIC_NON_NEGATIVE_COORDINATE:
                    if (((symmetry_type in "vV") and (point[0] < 0)) or
                        ((symmetry_type in "hH") and (point[1] < 0))):
                        filtered_out.add(point)
                        ##print("non_compliant with first-symmetric-to-be-placed-with-positive-coordinate rule: ", point)
                                
        allowed_points -= filtered_out

        # handles symmetry after everything else so not to false-negative previous tests:
        # if the vertex has a user-defined symmetric already placed at (x,y), only the
        # grid point (-x,y) --- or (x,-y), depending on the symmetry type --- will be considered
        if symmetric_vertex not in [None, vertex]:
            symmetric_vertex_location = coordinates[symmetric_vertex] 
            if symmetric_vertex_location != None:
                if symmetry_type in "vV":
                # vertical symmetry
                    forced_point = (-symmetric_vertex_location[0], symmetric_vertex_location[1])
                else:
                # horizontal symmetry
                    forced_point = (symmetric_vertex_location[0], -symmetric_vertex_location[1])

                if forced_point in allowed_points:
                    allowed_points.clear()
                    allowed_points.add(forced_point)
                else:
                    allowed_points.clear()
                    
    result_list = list(allowed_points)

## ENHANCEMENT:
##
##    if (len(model) > 0) and (len(result_list) > 0) and (result_list[0] == model[vertex]):
##        result_list == result_list[1:] + [result_list[0]] # avoids starting at the exact same point from model
        
    debug("result_list = " + spillguts(result_list), HIGH)
    return result_list

def mark_place_next_vertex_process(recursive_level: int, order_index: int):
    global place_next_vertex_counter
    global place_next_vertex_by_level_and_order_counter

    place_next_vertex_counter += 1
    while len(place_next_vertex_by_level_and_order_counter) <= recursive_level:
        place_next_vertex_by_level_and_order_counter.append([])
    while len(place_next_vertex_by_level_and_order_counter[recursive_level]) <= order_index:
        place_next_vertex_by_level_and_order_counter[recursive_level].append(0)
    place_next_vertex_by_level_and_order_counter[recursive_level][order_index] += 1

def place_vertices(graph, recursive_level, already_placed, progress, coordinates, model=set()):#, already_avoided_coincident=False):
    global collect_work_summary
## ENHANCEMENT:
##
##    if (OPTIMIZE_FOR_YES and (get_elapsed_time() > (MAX_TIME_YES_OPTIMIZATION * (recursive_level + 1)))) or \
##       ((not OPTIMIZE_FOR_YES) and (get_elapsed_time() > MAX_TIME_GLOBAL) and (MAX_TIME_GLOBAL >= 0)):
##        graph[COORDINATES][0] = TIMEOUT
##        return False

    if get_elapsed_time() > TIME_LIMIT:
        graph[COORDINATES][0] = TIMEOUT
        return False

    n = graph[NUMBER_OF_VERTICES]
    placement_order = graph[PLACEMENT_ORDER]
    predescessors = graph[PREDESCESSORS]
    vertex = placement_order[already_placed]
    k = MIN_K * 2**recursive_level
    radius = k * HALF_OF_GRAY_AREA_THICKNESS

    possible_locations = determine_possible_locations(graph, already_placed, coordinates[:], radius, model)

    if len(possible_locations) == 0:
    ##if (already_placed > 0) and (len(possible_locations) < 4):
        return False # the given model cannot be refined any further

    if (already_placed == n-1):   
        # if this is the last vertex to be placed, prepares the new model to be passed along
        new_model_list = [None]
        for coordinate in coordinates[1:]:
            if coordinate != None:
                new_coordinate = (2*coordinate[0], 2*coordinate[1])
                new_model_list += [new_coordinate]
            else:
                new_model_list += [None]

    reset_current_progress_level(progress, n, recursive_level, already_placed, len(possible_locations))
    
##    if not already_avoided_coincident:
##        for i in range(already_placed):
##            v = placement_order[i]
##            if coordinates[v] in possible_locations:
##                possible_locations.remove(coordinates[v])
##                already_avoided_coincident = True

    for location in possible_locations[-1::-1]:

        if collect_work_summary:
            mark_place_next_vertex_process(recursive_level, already_placed)

        coordinates[vertex] = location        

        if DISPLAY_PROGRESS:
            increment_progress_total_attempts(progress)
            increment_progress_level(progress, n, recursive_level, already_placed)
            if progress[0] % PROGRESS_STEP == 0:
                if not OPTIMIZE_FOR_YES:
                    print()
                    print("current (eps=7/%d):" % (10*k), coordinates[1:])
                    if graph[COORDINATES][0] > 0:
                        print("best (at eps=7/%d):" % (10*graph[COORDINATES][0]), graph[COORDINATES][1:])
                display_progress(progress, n, recursive_level, already_placed)

        if already_placed == n-1:
            # when placing the last vertex...
            
            if k > graph[COORDINATES][0]:
                coordinates[0] = k # stores the parameter k of the most refined trigraph embodiment
                graph[COORDINATES] = coordinates[:]

            if OPTIMIZE_FOR_YES or is_UDG_model(coordinates, graph, radius): # found UDG (non-sandwich!) model
                coordinates[0] = k # stores the parameter k of the conclusive UDG model
                graph[COORDINATES] = coordinates[:]
                if DISPLAY_PROGRESS:
                    display_progress(progress, n, recursive_level, already_placed)
                return True

            # recurses
            if recursive_level < MAX_RECURSIVE_LEVELS:

                debug(coordinates)

                new_model_list += [(2*location[0], 2*location[1])]
                new_model = tuple(new_model_list)
                if place_vertices(graph, recursive_level + 1, 0, progress, [coordinates[0]] + [None]*n, new_model):
                    return True
                elif graph[COORDINATES][0] == MIN_K * 2**MAX_RECURSIVE_LEVELS:
                    # found an inconclusive model in the most refined granularity
                    if OPTIMIZE_FOR_NO: 
                        return True  # stop immediately when optimizing for a NO answer
                                      # (a NO certificate won't be found anyway)
                del(new_model_list[-1])

            if OPTIMIZE_FOR_NO:
                return False  # won't look any further in this granularity after a trigraph model
                              # has been found here, if one wants to optimize for a NO answer
                              # (if there is a certificate, it must be found at refined granularities anyway)
        else:
            # when placing an intermediary vertex (i.e. not the last one)...
            if place_vertices(graph, recursive_level, already_placed + 1, progress, coordinates[:], model):
                return True
            elif graph[COORDINATES][0] == k:
                if OPTIMIZE_FOR_NO:
                    return False  # found an inconclusive model in the refined granularity,
                                  # should not look any further when optimizing for a NO answer
                    
            
    coordinates[vertex] = None

    return False



def main() -> None:
    # abcdefghijklmnopqrstuvwxyz
    # -b---f-hij--mno-qr-t--wxyz
    parser = argparse.ArgumentParser(description="Check if a graph is a Unit Disk Graph (UDG).")
    group = parser.add_mutually_exclusive_group(required=True)
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
        "-a", "--order", type=str, default="bfs",
        help="Vertex placement order (comma-separated; or 'bfs:root' for BFS, with optional root, by default vertex with fewer neighbors; 'c' for crescent)")
    parser.add_argument(
        "-k", "--init_gran", type=int, default=1,
        help="Initial granularity (type k for eps=0.7/k, default 1)")
    parser.add_argument(
        "-l", "--max_levels", type=int, default=3,
        help="Max levels deep (type 1 for no granularity refinement at all), default 3)")
    parser.add_argument(
        "-s", "--print_work_summary", action="store_true",
        help="Print a summary of work done")
    parser.add_argument(
        "graph", metavar="GRAPH", nargs="?", default="",
        help="Input graph description")

    args = parser.parse_args()


    if args.graph6:
        g = Graph6Converter.g6_to_graph(args.graph)
    elif args.edge_list:
        g = Graph6Converter.edge_list_to_graph(args.graph)
    else:
        print("Error: No input format specified.")
        return
    
    G = create_empty_graph(g.number_of_nodes())

    for e in g.edges():
        add_edge(G, (e[0] + 1, e[1] + 1))  # converting from 0-based to 1-based indexing
        
    debug("edges = " + spillguts(G[NEIGHBORS]))

    set_placement_order(G, args.order)

    if args.print_work_summary:
        global collect_work_summary
        collect_work_summary = True

    if False:  # interactive mode
        read_ordered_groups(G)
        read_fixed_sectors(G)
        read_symmetric_vertices(G)
        read_placement_options(G)

    MIN_K = args.init_gran

    MAX_RECURSIVE_LEVELS = args.max_levels - 1

    global DISPLAY_PROGRESS
    DISPLAY_PROGRESS = args.verbose

    progress = initialize_progress()
    
    info("\n-------------------------")
    info("\nWorking...")

    result = INCONCLUSIVE
    result_msg = ""
    elapsed_time = 0
    global START_TIME
    START_TIME = time.time()

    if MAX_TIME_YES_OPTIMIZATION != 0:
        global OPTIMIZE_FOR_YES
        OPTIMIZE_FOR_YES = True
     
        recursive_level = 0
        while recursive_level <= MAX_RECURSIVE_LEVELS:
            global TIME_LIMIT
            TIME_LIMIT = get_elapsed_time() + MAX_TIME_YES_OPTIMIZATION
            if MAX_TIME_GLOBAL > 0:
                TIME_LIMIT = min(TIME_LIMIT, MAX_TIME_GLOBAL)

            if place_vertices(G, recursive_level, 0, progress, G[COORDINATES][:]):
                if is_UDG_model(G[COORDINATES], G,
                                HALF_OF_GRAY_AREA_THICKNESS * G[COORDINATES][0]):
                    # an UDG model (non-sandwich) was found!
                    result_msg += obtain_printable_model(G, True)
                    result_msg += "\n\nResult: UDG."
                    result = YES
                    break
                
            recursive_level += 1
            clear_coordinates(G)
                
    if result == INCONCLUSIVE:

        OPTIMIZE_FOR_YES = False
        TIME_LIMIT = MAX_TIME_GLOBAL
        clear_coordinates(G)

        if place_vertices(G, 0, 0, progress, G[COORDINATES][:]):
            if is_UDG_model(G[COORDINATES], G,
                            HALF_OF_GRAY_AREA_THICKNESS * G[COORDINATES][0]):
                # an UDG model (non-sandwich) was found!
                result_msg += obtain_printable_model(G, True)
                result_msg += "\n\nResult: UDG."
                result = YES
            else:
                result_msg += obtain_printable_model(G, False)
                result_msg += "\n\nThe program found a trigraph realization at the thinnest allowed granularity."
                result_msg += "\nThus, a cerficate for a NO answer could not be found."
                result_msg += "\n\nResult: INCONCLUSIVE."
                result = INCONCLUSIVE

        elif G[COORDINATES][0] == TIMEOUT:
            result_msg += "\n\nThe program reached the maximum allowed run time."
            result_msg += "\n\nResult: INCONCLUSIVE."
            result = INCONCLUSIVE

        elif G[COORDINATES][0] in [0, MIN_K * 2**MAX_RECURSIVE_LEVELS]:
            result_msg += obtain_printable_model(G, False)
            result_msg += "\n\nThe program reached the maximum intended granularity."
            result_msg += "\nIt went through to granularity parameter k = %d \nand (still) " + \
                          "a trigraph realization -- but no UDG model! -- was found." % G[COORDINATES][0]
            result_msg += "\n\nResult: INCONCLUSIVE."
            result = INCONCLUSIVE
            
        else:
            # no UDG sandwich model was found for a certain K factor
            result_msg += "\n\nThere are no sandwich UDG models (trigraph embodiments) in the k=%d granularity." % (max(MIN_K, G[COORDINATES][0] * 2))
            result_msg += "\n\nResult: NOT UDG."
            result = NO

    elapsed_time = get_elapsed_time()
    result_msg += "\n(Took " + format_time(elapsed_time, True) + ".)"

    if args.print_work_summary:
        print(f"Placement order: {G[PLACEMENT_ORDER]}")
        print(f"Total place_next_vertex calls: {place_next_vertex_counter}")
        print(f"  lvl:ord  calls")
        for level, counters in enumerate(place_next_vertex_by_level_and_order_counter):
            for order, count in enumerate(counters):
                print(f"  {level:3d}:{order:3d}  {count:9,d}".replace(',', ' '))

    print(result_msg)
    print("\n=====================================\n")
    if args.print_vertex:
        print("\nVertex coordinates:")
        n = G[NUMBER_OF_VERTICES]
        for i in range(1, n+1):
            coordinate = G[COORDINATES][i]
            print("Vertex %d: %s" % (i, spillguts(coordinate)))

if __name__ == "__main__":
    main()

