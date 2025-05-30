import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cvxpy as cp

def generate_random_convex_polygon_method1(n_points):
    """Generate points on a circle with random radii to create more varied polygons"""
    angles = np.sort(np.random.uniform(0, 2*np.pi, n_points))
    # Use random radii but ensure they're not too small to avoid degenerate cases
    radii = np.random.uniform(2, 5, n_points)
    
    points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    # Translate to positive quadrant
    points += 5
    
    hull = ConvexHull(points)
    polygon = points[hull.vertices]
    return polygon

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
    
    hull = ConvexHull(points)
    polygon = points[hull.vertices]
    return polygon

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
    angle_variation = np.random.uniform(-0.3, 0.3, n_points)
    angles = base_angles + angle_variation
    
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
    
    hull = ConvexHull(points)
    polygon = points[hull.vertices]
    return polygon

def polygon_to_Ab(polygon):
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
    # John/Löwner ellipse via convex optimization
    A, b = polygon_to_Ab(polygon)
    C = cp.Variable((2,2), symmetric=True)
    d = cp.Variable(2)
    constraints = [C >> 0]
    for i in range(A.shape[0]):
        constraints.append(cp.norm(C @ A[i], 2) + A[i] @ d <= b[i])
    prob = cp.Problem(cp.Maximize(cp.log_det(C)), constraints)
    prob.solve()
    return C.value, d.value

def ellipse_points(C, d, num=200):
    t = np.linspace(0, 2*np.pi, num)
    circle = np.stack([np.cos(t), np.sin(t)])
    ell = (C @ circle) + d[:,None]
    return ell[0], ell[1]

def plot_polygon_and_ellipse(polygon, C, d, method_name=""):
    plt.figure(figsize=(8,8))
    plt.plot(*polygon.T, 'b-', linewidth=2, label='Polygon')
    plt.plot([polygon[-1,0], polygon[0,0]], [polygon[-1,1], polygon[0,1]], 'b-', linewidth=2)
    
    # Plot vertices as points
    plt.scatter(polygon[:, 0], polygon[:, 1], c='blue', s=50, zorder=5)
    
    x, y = ellipse_points(C, d)
    plt.plot(x, y, 'r-', linewidth=2, label='Max Area Inscribed Ellipse')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f'Convex Polygon with Maximum Area Inscribed Ellipse\n{method_name}')
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
    
    print(f"\nGenerated polygon with {len(polygon)} vertices")
    
    try:
        C, d = find_max_inscribed_ellipse(polygon)
        plot_polygon_and_ellipse(polygon, C, d, method_name)
        print("\nEllipse center:", d)
        print("Ellipse matrix C (shape):\n", C)
        print("Ellipse area:", np.pi * np.linalg.det(C))
    except Exception as e:
        print(f"Error finding inscribed ellipse: {e}")
        # Plot just the polygon
        plt.figure(figsize=(8,8))
        plt.plot(*polygon.T, 'b-', linewidth=2, label='Polygon')
        plt.plot([polygon[-1,0], polygon[0,0]], [polygon[-1,1], polygon[0,1]], 'b-', linewidth=2)
        plt.scatter(polygon[:, 0], polygon[:, 1], c='blue', s=50, zorder=5)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(f'Generated Polygon\n{method_name}')
        plt.show()

if __name__ == "__main__":
    main()