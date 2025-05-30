import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import cvxpy as cp

def generate_random_convex_polygon_method1(n_points):
    """Generate points on a circle with random radii to create more varied polygons"""
    # Generate points in polar coordinates to ensure we get exactly n_points vertices
    angles = np.sort(np.random.uniform(0, 2*np.pi, n_points))
    # Use random radii but ensure they're not too small to avoid degenerate cases
    radii = np.random.uniform(2, 5, n_points)
    
    points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    # Translate to positive quadrant
    points += 5
    
    # Since points are generated in sorted angular order, they already form a convex polygon
    # No need for ConvexHull - just return the points in order
    return points

def generate_random_convex_polygon_method2(n_points):
    """Generate points in an annulus (ring) to avoid clustering"""
    angles = np.sort(np.random.uniform(0, 2*np.pi, n_points))
    # Generate radii in an annulus to avoid points too close to center
    inner_radius = 2
    outer_radius = 5
    radii = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, n_points))
    
    points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    # Translate to positive quadrant
    points += 6
    
    # Since points are generated in sorted angular order, they form a convex polygon
    return points

def generate_random_convex_polygon_method3(n_points):
    """Generate points by sampling from a larger set and selecting diverse ones"""
    # Generate many more points than needed
    candidate_points = np.random.rand(n_points * 5, 2) * 10
    
    # Find convex hull of all candidates
    hull = ConvexHull(candidate_points)
    hull_vertices = candidate_points[hull.vertices]
    
    # If we have enough vertices, randomly select n_points from them
    if len(hull_vertices) >= n_points:
        selected_indices = np.random.choice(len(hull_vertices), n_points, replace=False)
        selected_points = hull_vertices[selected_indices]
    else:
        # If not enough, use all hull vertices and add some interior points
        remaining = n_points - len(hull_vertices)
        interior_points = np.random.rand(remaining, 2) * 10
        selected_points = np.vstack([hull_vertices, interior_points])
    
    # Create final hull
    final_hull = ConvexHull(selected_points)
    polygon = selected_points[final_hull.vertices]
    return polygon

def generate_random_convex_polygon_method4(n_points):
    """Generate a more star-like polygon using variable radii at fixed angles"""
    # Create evenly spaced angles with some random variation
    base_angles = np.linspace(0, 2*np.pi, n_points + 1)[:-1]  # Remove duplicate 2π
    angle_variation = np.random.uniform(-0.2, 0.2, n_points)
    angles = (base_angles + angle_variation) % (2*np.pi)  # Keep angles in [0, 2π]
    angles = np.sort(angles)  # Sort to maintain convexity
    
    # Generate radii with more variation to create star-like appearance
    base_radius = 3
    radii = base_radius + np.random.uniform(-1.5, 2.5, n_points)
    # Ensure no negative radii
    radii = np.maximum(radii, 0.5)
    
    points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    # Translate to positive quadrant
    points += 5
    
    # Points are already in convex order, no need for ConvexHull
    return points

def find_inner_convex_hull(polygon, num_candidate_points=1000):
    """
    Find the inner convex hull of a starry polygon by generating candidate points
    inside the polygon and finding their convex hull.
    """
    # Find the bounding box of the polygon
    min_x, min_y = np.min(polygon, axis=0)
    max_x, max_y = np.max(polygon, axis=0)
    
    # Generate candidate points within the bounding box
    candidates = []
    attempts = 0
    max_attempts = num_candidate_points * 10
    
    while len(candidates) < num_candidate_points and attempts < max_attempts:
        # Generate random point in bounding box
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = np.array([x, y])
        
        # Check if point is inside the polygon using ray casting
        if point_in_polygon(point, polygon):
            candidates.append(point)
        
        attempts += 1
    
    if len(candidates) < 3:
        print(f"Warning: Only found {len(candidates)} interior points. Using polygon centroid.")
        # Fallback: use centroid and some interior points
        centroid = np.mean(polygon, axis=0)
        return np.array([centroid])
    
    candidates = np.array(candidates)
    
    # Find convex hull of interior points
    if len(candidates) >= 3:
        hull = ConvexHull(candidates)
        inner_polygon = candidates[hull.vertices]
        return inner_polygon
    else:
        return candidates

def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def find_inner_convex_hull_optimized(polygon, method='inscribed_circle'):
    """
    Find inner convex hull using different optimization strategies.
    """
    if method == 'inscribed_circle':
        return find_inner_hull_via_inscribed_circle(polygon)
    elif method == 'shrinking':
        return find_inner_hull_via_shrinking(polygon)
    else:
        return find_inner_convex_hull(polygon)

def find_inner_hull_via_inscribed_circle(polygon):
    """
    Find inner convex hull by first finding the largest inscribed circle,
    then generating points around it.
    """
    # Find approximate center (centroid)
    center = np.mean(polygon, axis=0)
    
    # Find the maximum radius circle that fits inside
    min_distance_to_edge = float('inf')
    
    # Check distance from center to each edge of the polygon
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        
        # Distance from point to line segment
        distance = point_to_line_distance(center, p1, p2)
        min_distance_to_edge = min(min_distance_to_edge, distance)
    
    # Create inner polygon as a circle with radius slightly smaller than max inscribed circle
    radius = min_distance_to_edge * 0.8  # Use 80% of max radius for safety
    n_points = max(8, len(polygon) // 4)  # Reasonable number of points for inner hull
    
    angles = np.linspace(0, 2*np.pi, n_points + 1)[:-1]
    inner_points = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    
    return inner_points

def point_to_line_distance(point, line_start, line_end):
    """Calculate the shortest distance from a point to a line segment."""
    # Vector from line_start to line_end
    line_vec = line_end - line_start
    # Vector from line_start to point
    point_vec = point - line_start
    
    # Project point_vec onto line_vec
    line_length_sq = np.dot(line_vec, line_vec)
    if line_length_sq == 0:
        return np.linalg.norm(point_vec)
    
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_length_sq))
    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)

def find_inner_hull_via_shrinking(polygon):
    """
    Find inner convex hull by shrinking the polygon towards its center.
    This is a simplified approach - creates a smaller version of the polygon.
    """
    center = np.mean(polygon, axis=0)
    # Shrink polygon towards center by 60%
    shrink_factor = 0.6
    inner_points = center + shrink_factor * (polygon - center)
    
    # Take every few points to create a simpler inner hull
    step = max(1, len(inner_points) // 12)  # Limit to ~12 points
    simplified_inner = inner_points[::step]
    
    return simplified_inner

def polygon_to_Ab(polygon):
    """Returns A, b such that Ax <= b describes the polygon"""
    n = len(polygon)
    A = []
    b = []
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i+1)%n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        # Handle zero-length edges
        norm_length = np.linalg.norm(normal)
        if norm_length > 1e-10:
            normal = normal / norm_length
            A.append(normal)
            b.append(np.dot(normal, p1))
    return np.array(A), np.array(b)
    # Returns A, b such that Ax <= b describes the polygon
    n = len(polygon)
    A = []
    b = []
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i+1)%n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        normal = normal / np.linalg.norm(normal)
        A.append(normal)
        b.append(np.dot(normal, p1))
    return np.array(A), np.array(b)

def find_max_inscribed_ellipse(polygon):
    """John/Löwner ellipse via convex optimization"""
    try:
        A, b = polygon_to_Ab(polygon)
        
        # Check if we have valid constraints
        if len(A) == 0:
            raise ValueError("No valid constraints found - polygon may be degenerate")
        
        C = cp.Variable((2,2), symmetric=True)
        d = cp.Variable(2)
        constraints = [C >> 0]
        
        for i in range(A.shape[0]):
            constraints.append(cp.norm(C @ A[i], 2) + A[i] @ d <= b[i])
        
        prob = cp.Problem(cp.Maximize(cp.log_det(C)), constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed with status: {prob.status}")
        
        if C.value is None or d.value is None:
            raise ValueError("Optimization returned None values")
        
        return C.value, d.value
        
    except Exception as e:
        print(f"Error in ellipse optimization: {e}")
        # Fallback: return a small circle at the centroid
        center = np.mean(polygon, axis=0)
        radius = 0.5
        C_fallback = np.eye(2) * radius**2
        return C_fallback, center

def ellipse_points(C, d, num=200):
    t = np.linspace(0, 2*np.pi, num)
    circle = np.stack([np.cos(t), np.sin(t)])
    ell = (C @ circle) + d[:,None]
    return ell[0], ell[1]

def plot_polygon_and_ellipse(polygon, inner_hull, C, d, method_name=""):
    plt.figure(figsize=(10,10))
    
    # Plot outer starry polygon
    plt.plot(*polygon.T, 'b-', linewidth=2, label='Starry Polygon', alpha=0.7)
    plt.plot([polygon[-1,0], polygon[0,0]], [polygon[-1,1], polygon[0,1]], 'b-', linewidth=2, alpha=0.7)
    plt.scatter(polygon[:, 0], polygon[:, 1], c='blue', s=30, alpha=0.7, zorder=5)
    
    # Plot inner convex hull
    plt.plot(*inner_hull.T, 'g-', linewidth=2, label='Inner Convex Hull')
    plt.plot([inner_hull[-1,0], inner_hull[0,0]], [inner_hull[-1,1], inner_hull[0,1]], 'g-', linewidth=2)
    plt.scatter(inner_hull[:, 0], inner_hull[:, 1], c='green', s=50, zorder=5)
    
    # Plot inscribed ellipse
    if C is not None and d is not None:
        x, y = ellipse_points(C, d)
        plt.plot(x, y, 'r-', linewidth=3, label='Max Area Inscribed Ellipse')
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f'Starry Polygon with Inner Convex Hull and Inscribed Ellipse\n{method_name}')
    plt.show()

def main():
    n_points = int(input("Enter the number of points for the convex polygon: "))
    
    print("\nChoose generation method:")
    print("1. Random radii on circle (good for star-like shapes)")
    print("2. Points in annulus (avoids center clustering)")
    print("3. Select from larger candidate set")
    print("4. Semi-regular with variation (most star-like)")
    print("5. Original method (for comparison)")
    
    method = input("Enter method number (1-5): ").strip()
    
    if method == "1":
        polygon = generate_random_convex_polygon_method1(n_points)
        method_name = "Method 1: Random radii on circle"
    elif method == "2":
        polygon = generate_random_convex_polygon_method2(n_points)
        method_name = "Method 2: Points in annulus"
    elif method == "3":
        polygon = generate_random_convex_polygon_method3(n_points)
        method_name = "Method 3: Selection from candidates"
    elif method == "4":
        polygon = generate_random_convex_polygon_method4(n_points)
        method_name = "Method 4: Semi-regular with variation"
    else:
        # Original method
        points = np.random.rand(n_points, 2) * 10
        hull = ConvexHull(points)
        polygon = points[hull.vertices]
        method_name = "Original method"
    
    print(f"\nGenerated starry polygon with {len(polygon)} vertices")
    
    # Find inner convex hull
    print("Finding inner convex hull...")
    print("\nChoose inner hull method:")
    print("1. Random points inside polygon")
    print("2. Inscribed circle approach (faster)")
    print("3. Shrinking approach")
    
    inner_method = input("Enter inner hull method (1-3, default=2): ").strip() or "2"
    
    if inner_method == "1":
        inner_hull = find_inner_convex_hull(polygon, num_candidate_points=2000)
        inner_method_name = "Random interior points"
    elif inner_method == "3":
        inner_hull = find_inner_convex_hull_optimized(polygon, method='shrinking')
        inner_method_name = "Shrinking approach"
    else:
        inner_hull = find_inner_hull_via_inscribed_circle(polygon)
        inner_method_name = "Inscribed circle approach"
    
    print(f"Found inner convex hull with {len(inner_hull)} vertices using {inner_method_name}")
    
    try:
        # Find inscribed ellipse in the inner convex hull
        C, d = find_max_inscribed_ellipse(inner_hull)
        plot_polygon_and_ellipse(polygon, inner_hull, C, d, f"{method_name} + {inner_method_name}")
        
        print(f"\nResults:")
        print(f"Starry polygon vertices: {len(polygon)}")
        print(f"Inner convex hull vertices: {len(inner_hull)}")
        print(f"Ellipse center: {d}")
        print(f"Ellipse area: {np.pi * np.linalg.det(C):.4f}")
        
    except Exception as e:
        print(f"Error finding inscribed ellipse: {e}")
        # Plot just the polygons
        plt.figure(figsize=(10,10))
        plt.plot(*polygon.T, 'b-', linewidth=2, label='Starry Polygon', alpha=0.7)
        plt.plot([polygon[-1,0], polygon[0,0]], [polygon[-1,1], polygon[0,1]], 'b-', linewidth=2, alpha=0.7)
        plt.scatter(polygon[:, 0], polygon[:, 1], c='blue', s=30, alpha=0.7)
        
        plt.plot(*inner_hull.T, 'g-', linewidth=2, label='Inner Convex Hull')
        plt.plot([inner_hull[-1,0], inner_hull[0,0]], [inner_hull[-1,1], inner_hull[0,1]], 'g-', linewidth=2)
        plt.scatter(inner_hull[:, 0], inner_hull[:, 1], c='green', s=50)
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(f'Starry Polygon with Inner Convex Hull\n{method_name} + {inner_method_name}')
        plt.show()

if __name__ == "__main__":
    main()