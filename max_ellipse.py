import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import cvxpy as cp

def generate_random_convex_polygon_method(n_points):
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
    else:
        return find_inner_convex_hull(polygon)

def find_inner_hull_via_inscribed_circle(polygon):
    """
    Find inner convex hull by first finding the largest inscribed circle,
    then generating points around it, and finally creating a tight inner hull
    by removing points outside boundaries.
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
    
    # Create initial inner polygon as a circle with radius slightly smaller than max inscribed circle
    radius = min_distance_to_edge * 0.8  # Use 80% of max radius for safety
    n_points = max(16, len(polygon) // 2)  # More points for better precision
    
    angles = np.linspace(0, 2*np.pi, n_points + 1)[:-1]
    inner_points = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Create a tighter inner hull by removing points outside boundaries
    tight_hull_points = []
    for point in inner_points:
        # Check if point is inside the polygon
        if point_in_polygon(point, polygon):
            tight_hull_points.append(point)
    
    if len(tight_hull_points) < 3:
        # If we don't have enough points, fall back to the original circle points
        return inner_points
    
    # Convert to numpy array and find convex hull
    tight_hull_points = np.array(tight_hull_points)
    hull = ConvexHull(tight_hull_points)
    tight_hull = tight_hull_points[hull.vertices]
    
    return tight_hull

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


def find_max_inscribed_ellipse(polygon):
    """John/Löwner ellipse via convex optimization"""
    try:
        A, b = polygon_to_Ab(polygon)
        
        # Check if we have valid constraints
        if len(A) == 0:
            raise ValueError("No valid constraints found - polygon may be degenerate")
        
        C = cp.Variable((2,2), symmetric=True)
        d = cp.Variable(2)
        constraints = [C >> 0]  # C must be positive semidefinite
        
        for i in range(A.shape[0]):
            constraints.append(cp.norm(C @ A[i], 2) + A[i] @ d <= b[i])
        
        # Add constraint to ensure ellipse is not too small
        min_area = 0.1  # Minimum area constraint
        constraints.append(cp.log_det(C) >= np.log(min_area))
        
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

def is_star_shaped(polygon, point):
    """Check if a polygon is visible from a given point."""
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        # Check if the line segment from point to any vertex intersects any edge
        for j in range(n):
            p3 = polygon[j]
            p4 = polygon[(j + 1) % n]
            if i != j and line_segments_intersect(point, p1, p3, p4):
                return False
    return True

def line_segments_intersect(p1, p2, p3, p4):
    """Check if two line segments intersect."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def find_kernel(polygon, num_candidates=1000):
    """Find a point in the kernel of a star-shaped polygon."""
    # Start with the centroid
    center = np.mean(polygon, axis=0)
    
    if is_star_shaped(polygon, center):
        return center
    
    # If centroid is not in kernel, try to find a point that is
    min_x, min_y = np.min(polygon, axis=0)
    max_x, max_y = np.max(polygon, axis=0)
    
    for _ in range(num_candidates):
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = np.array([x, y])
        
        if is_star_shaped(polygon, point):
            return point
    
    # If no point found, return None
    return None

def find_max_inscribed_ellipse_star(polygon):
    """Find the maximum inscribed ellipse in a star-shaped polygon."""
    # First, find a point in the kernel
    kernel_point = find_kernel(polygon)
    if kernel_point is None:
        raise ValueError("The polygon is not star-shaped or no kernel point could be found")
    
    # Transform the polygon to make the kernel point the origin
    transformed_polygon = polygon - kernel_point
    
    # Find the maximum inscribed ellipse in the transformed space
    C, d = find_max_inscribed_ellipse(transformed_polygon)
    
    # Transform back: the ellipse center in the original system is d + kernel_point
    d = d + kernel_point
    
    return C, d