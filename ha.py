#!/usr/bin/env python3
"""
Hexagonal/Triangular Grid Analyzer (ha.py)
Draws equilateral triangle grid with hexagonal patterns and circle analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PatchCollection
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Set
import math


@dataclass(slots=True)
class Point:
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass(slots=True)
class Triangle:
    p1: Point
    p2: Point 
    p3: Point
    
    def center(self) -> Point:
        return Point(
            (self.p1.x + self.p2.x + self.p3.x) / 3,
            (self.p1.y + self.p2.y + self.p3.y) / 3
        )
    
    def vertices(self) -> List[Point]:
        return [self.p1, self.p2, self.p3]


@dataclass(slots=True) 
class Hexagon:
    center: Point
    vertices: List[Point]
    
    def triangles(self) -> List[Triangle]:
        """Return 6 triangles that make up this hexagon."""
        triangles = []
        for i in range(6):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % 6]
            triangles.append(Triangle(self.center, v1, v2))
        return triangles


class HexagonalGrid:
    def __init__(self, side_length: float = 1.0, grid_size: int = 5, center_x: float = 0.0, center_y: float = 0.0, radius: float = 2.0):
        self.side_length = side_length
        self.grid_size = grid_size
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.height = side_length * math.sqrt(3) / 2  # Height of equilateral triangle
        
        # Calculate grid bounds based on center and radius
        self.area_size = int(radius + 1) + 2  # r+1 plus some margin
        
        self.vertices: Set[Tuple[float, float]] = set()
        self.triangles: List[Triangle] = []
        self.hexagons: List[Hexagon] = []
        
        self._generate_grid()
    
    def _generate_grid(self):
        """Generate triangular grid with hexagonal patterns."""
        # Generate triangular lattice vertices centered around the specified point
        self.vertices = set()
        
        # Create triangular lattice covering area (radius + 1) around center
        for i in range(-self.area_size, self.area_size + 1):
            for j in range(-self.area_size, self.area_size + 1):
                # Basic triangular lattice
                x = j * self.side_length + (i % 2) * self.side_length * 0.5 + self.center_x
                y = i * self.height + self.center_y
                
                # Only include vertices within reasonable distance from center
                dist_from_center = math.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
                if dist_from_center <= self.radius + 2:  # Include some margin
                    self.vertices.add((x, y))
        
        # Generate triangles from the lattice
        vertices_list = list(self.vertices)
        self.triangles = []
        
        # Simple approach: for each vertex, find two other vertices that form equilateral triangle
        processed_triangles = set()
        
        for i, v1 in enumerate(vertices_list):
            for j, v2 in enumerate(vertices_list[i+1:], i+1):
                for k, v3 in enumerate(vertices_list[j+1:], j+1):
                    # Check if these three vertices form an equilateral triangle
                    d12 = math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
                    d13 = math.sqrt((v1[0] - v3[0])**2 + (v1[1] - v3[1])**2)
                    d23 = math.sqrt((v2[0] - v3[0])**2 + (v2[1] - v3[1])**2)
                    
                    # Check if all sides are approximately equal to side_length
                    tolerance = 0.1
                    if (abs(d12 - self.side_length) < tolerance and 
                        abs(d13 - self.side_length) < tolerance and 
                        abs(d23 - self.side_length) < tolerance):
                        
                        # Create triangle
                        triangle = Triangle(
                            Point(v1[0], v1[1]),
                            Point(v2[0], v2[1]),
                            Point(v3[0], v3[1])
                        )
                        
                        # Avoid duplicates
                        triangle_key = tuple(sorted([(v1[0], v1[1]), (v2[0], v2[1]), (v3[0], v3[1])]))
                        if triangle_key not in processed_triangles:
                            processed_triangles.add(triangle_key)
                            self.triangles.append(triangle)
        
        # Generate hexagons (simplified approach)
        self._generate_hexagons()
    
    def _triangle_exists(self, new_triangle: Triangle) -> bool:
        """Check if triangle already exists (accounting for vertex order)."""
        new_vertices = {(t.x, t.y) for t in new_triangle.vertices()}
        for existing in self.triangles:
            existing_vertices = {(t.x, t.y) for t in existing.vertices()}
            if new_vertices == existing_vertices:
                return True
        return False
    
    def _generate_hexagons(self):
        """Generate hexagons from the triangular grid."""
        # Find hexagon centers (vertices that have 6 triangle neighbors)
        vertex_to_triangles = {}
        
        for triangle in self.triangles:
            for vertex in triangle.vertices():
                key = (vertex.x, vertex.y)
                if key not in vertex_to_triangles:
                    vertex_to_triangles[key] = []
                vertex_to_triangles[key].append(triangle)
        
        # Find vertices with exactly 6 adjacent triangles (hexagon centers)
        for vertex_pos, adjacent_triangles in vertex_to_triangles.items():
            if len(adjacent_triangles) == 6:
                center = Point(vertex_pos[0], vertex_pos[1])
                
                # Find hexagon vertices (opposite vertices of adjacent triangles)
                hex_vertices = []
                for triangle in adjacent_triangles:
                    for vertex in triangle.vertices():
                        if abs(vertex.x - center.x) > 0.01 or abs(vertex.y - center.y) > 0.01:
                            # This is not the center vertex
                            dist = center.distance_to(vertex)
                            if abs(dist - self.side_length) < 0.01:
                                hex_vertices.append(vertex)
                
                # Remove duplicates and sort by angle
                unique_vertices = []
                for v in hex_vertices:
                    if not any(abs(v.x - uv.x) < 0.01 and abs(v.y - uv.y) < 0.01 for uv in unique_vertices):
                        unique_vertices.append(v)
                
                if len(unique_vertices) == 6:
                    # Sort vertices by angle around center
                    unique_vertices.sort(key=lambda v: math.atan2(v.y - center.y, v.x - center.x))
                    self.hexagons.append(Hexagon(center, unique_vertices))


class CircleAnalyzer:
    def __init__(self, grid: HexagonalGrid, circle_center: Point, radius: float):
        self.grid = grid
        self.circle_center = circle_center
        self.radius = radius
        
    def classify_triangles(self) -> Tuple[List[Triangle], List[Triangle], List[Triangle]]:
        """Classify triangles as inside, outside, or intersecting the circle."""
        inside = []
        outside = []
        intersecting = []
        
        for triangle in self.grid.triangles:
            classification = self._classify_triangle(triangle)
            if classification == "inside":
                inside.append(triangle)
            elif classification == "outside":
                outside.append(triangle)
            else:
                intersecting.append(triangle)
        
        return inside, outside, intersecting
    
    def classify_hexagons(self) -> Tuple[List[Hexagon], List[Hexagon], List[Hexagon]]:
        """Classify hexagons as inside, outside, or intersecting the circle."""
        inside = []
        outside = []
        intersecting = []
        
        for hexagon in self.grid.hexagons:
            classification = self._classify_hexagon(hexagon)
            if classification == "inside":
                inside.append(hexagon)
            elif classification == "outside":
                outside.append(hexagon)
            else:
                intersecting.append(hexagon)
        
        return inside, outside, intersecting
    
    def _classify_triangle(self, triangle: Triangle) -> str:
        """Classify single triangle relative to circle."""
        vertices = triangle.vertices()
        distances = [self.circle_center.distance_to(v) for v in vertices]
        
        inside_count = sum(1 for d in distances if d < self.radius)
        outside_count = sum(1 for d in distances if d > self.radius)
        
        if inside_count == 3:
            return "inside"
        elif outside_count == 3:
            return "outside"
        else:
            return "intersecting"
    
    def _classify_hexagon(self, hexagon: Hexagon) -> str:
        """Classify single hexagon relative to circle.""" 
        if not hexagon.vertices or len(hexagon.vertices) == 0:
            return "outside"  # Default for malformed hexagons
            
        distances = [self.circle_center.distance_to(v) for v in hexagon.vertices]
        
        inside_count = sum(1 for d in distances if d < self.radius)
        outside_count = sum(1 for d in distances if d > self.radius)
        
        if inside_count == len(hexagon.vertices):
            return "inside"
        elif outside_count == len(hexagon.vertices):
            return "intersecting"


class GridVisualizer:
    def __init__(self, grid: HexagonalGrid):
        self.grid = grid
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_aspect('equal')
        
    def draw_grid(self):
        """Draw the basic triangular grid."""
        print(f"Drawing {len(self.grid.triangles)} triangles...")
        
        # Draw triangle edges as individual lines
        for triangle in self.grid.triangles:
            vertices = triangle.vertices()
            # Draw each edge of the triangle
            for i in range(3):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % 3]
                self.ax.plot([v1.x, v2.x], [v1.y, v2.y], 'k-', linewidth=1.5, alpha=0.8)
        
        # Draw vertices
        if len(self.grid.vertices) < 100:  # Only show vertices for smaller grids
            for vertex_pos in self.grid.vertices:
                self.ax.plot(vertex_pos[0], vertex_pos[1], 'ko', markersize=4, alpha=0.8)
    
    def draw_hexagons(self):
        """Draw hexagons with thick lines."""
        for hexagon in self.grid.hexagons:
            hex_patch = patches.Polygon(
                [(v.x, v.y) for v in hexagon.vertices],
                fill=False,
                edgecolor='blue',
                linewidth=2.5,
                zorder=20
            )
            self.ax.add_patch(hex_patch)
    
    def draw_circle(self, center: Point, radius: float):
        """Draw circle."""
        circle = patches.Circle(
            (center.x, center.y),
            radius,
            fill=False,
            edgecolor='red',
            linewidth=3.0,
            zorder=30
        )
        self.ax.add_patch(circle)
        
        # Mark center
        self.ax.plot(center.x, center.y, 'ro', markersize=10, zorder=35)
    
    def highlight_shapes(self, inside_shapes, intersecting_shapes, shape_type="triangle"):
        """Highlight classified shapes."""
        # Inside shapes - green fill
        for shape in inside_shapes:
            if shape_type == "triangle":
                vertices = [(v.x, v.y) for v in shape.vertices()]
            else:  # hexagon
                vertices = [(v.x, v.y) for v in shape.vertices]
            
            patch = patches.Polygon(
                vertices,
                fill=True,
                facecolor='green',
                edgecolor='darkgreen',
                alpha=0.5,
                linewidth=2.0,
                zorder=10
            )
            self.ax.add_patch(patch)
        
        # Intersecting shapes - yellow fill
        for shape in intersecting_shapes:
            if shape_type == "triangle":
                vertices = [(v.x, v.y) for v in shape.vertices()]
            else:  # hexagon
                vertices = [(v.x, v.y) for v in shape.vertices]
            
            patch = patches.Polygon(
                vertices,
                fill=True,
                facecolor='yellow',
                edgecolor='darkorange',
                alpha=0.5,
                linewidth=2.0,
                zorder=10
            )
            self.ax.add_patch(patch)
    
    def show(self, title="Hexagonal Grid Analysis"):
        """Show the plot."""
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            patches.Patch(color='lightgreen', label='Inside circle'),
            patches.Patch(color='lightyellow', label='Intersecting circle'),
            patches.Patch(color='white', label='Outside circle'),
            patches.Patch(color='blue', label='Hexagons'),
            patches.Patch(color='red', label='Circle')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()


def find_nearest_vertex(grid: HexagonalGrid, x: float, y: float) -> Point:
    """Find nearest grid vertex to given coordinates."""
    min_dist = float('inf')
    nearest = None
    
    for vertex_pos in grid.vertices:
        dist = math.sqrt((x - vertex_pos[0])**2 + (y - vertex_pos[1])**2)
        if dist < min_dist:
            min_dist = dist
            nearest = Point(vertex_pos[0], vertex_pos[1])
    
    return nearest


def main():
    parser = argparse.ArgumentParser(description="Hexagonal/Triangular Grid Analyzer")
    parser.add_argument("-s", "--side", type=float, default=1.0, 
                       help="Side length of triangles (default: 1.0)")
    parser.add_argument("-g", "--grid-size", type=int, default=4,
                       help="Grid size (default: 4)")
    parser.add_argument("-r", "--radius", type=float, default=2.0,
                       help="Circle radius as multiple of side length (default: 2.0)")
    parser.add_argument("-x", "--center-x", type=float, default=0.0,
                       help="Circle center X coordinate (will snap to nearest vertex)")
    parser.add_argument("-y", "--center-y", type=float, default=0.0, 
                       help="Circle center Y coordinate (will snap to nearest vertex)")
    parser.add_argument("--shape", choices=["triangle", "hexagon"], default="triangle",
                       help="Shape type to analyze (default: triangle)")
    parser.add_argument("--no-hexagons", action="store_true",
                       help="Don't draw hexagon outlines")
    
    args = parser.parse_args()
    
    # Calculate radius first
    radius = args.radius * args.side
    
    # Create grid with proper coverage area
    print(f"Creating hexagonal grid (side={args.side}, center=({args.center_x:.2f}, {args.center_y:.2f}), coverage_radius={radius+1:.2f})...")
    grid = HexagonalGrid(side_length=args.side, grid_size=args.grid_size, 
                        center_x=args.center_x, center_y=args.center_y, radius=radius)
    
    # Debug info
    print(f"Generated: {len(grid.vertices)} vertices, {len(grid.triangles)} triangles, {len(grid.hexagons)} hexagons")
    
    # Use the specified center directly
    actual_center = Point(args.center_x, args.center_y)
    print(f"Circle: center=({actual_center.x:.2f}, {actual_center.y:.2f}), radius={radius:.2f}")
    
    # Analyze shapes
    analyzer = CircleAnalyzer(grid, actual_center, radius)
    
    if args.shape == "triangle":
        inside, outside, intersecting = analyzer.classify_triangles()
        print(f"Triangles: {len(inside)} inside, {len(intersecting)} intersecting, {len(outside)} outside")
    else:
        inside, outside, intersecting = analyzer.classify_hexagons()  
        print(f"Hexagons: {len(inside)} inside, {len(intersecting)} intersecting, {len(outside)} outside")
    
    # Visualize
    visualizer = GridVisualizer(grid)
    
    # Draw in proper order: grid first, then highlights, then hexagons and circle on top
    visualizer.draw_grid()
    visualizer.highlight_shapes(inside, intersecting, args.shape)
    
    if not args.no_hexagons:
        visualizer.draw_hexagons()
    
    visualizer.draw_circle(actual_center, radius)
    
    title = f"Hexagonal Grid - {args.shape.title()} Analysis (r={radius:.1f})"
    visualizer.show(title)


if __name__ == "__main__":
    main()