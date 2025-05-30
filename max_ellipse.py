import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cvxpy as cp

def generate_random_convex_polygon(n_points):
    points = np.random.rand(n_points, 2) * 10
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

def plot_polygon_and_ellipse(polygon, C, d):
    plt.figure(figsize=(8,8))
    plt.plot(*polygon.T, 'b-', label='Polygon')
    plt.plot([polygon[-1,0], polygon[0,0]], [polygon[-1,1], polygon[0,1]], 'b-')
    x, y = ellipse_points(C, d)
    plt.plot(x, y, 'r-', label='Max Area Inscribed Ellipse')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Convex Polygon with Maximum Area Inscribed Ellipse')
    plt.show()

def main():
    n_points = int(input("Enter the number of points for the convex polygon: "))
    polygon = generate_random_convex_polygon(n_points)
    C, d = find_max_inscribed_ellipse(polygon)
    plot_polygon_and_ellipse(polygon, C, d)
    print("\nEllipse center:", d)
    print("Ellipse matrix C (shape):\n", C)
    print("Ellipse area:", np.pi * np.linalg.det(C))

if __name__ == "__main__":
    main() 