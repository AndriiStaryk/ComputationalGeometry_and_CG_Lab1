import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from max_ellipse import (
    generate_random_convex_polygon_method,
    find_inner_hull_via_inscribed_circle,
    find_max_inscribed_ellipse,
    find_max_inscribed_ellipse_star,
    ellipse_points,
    is_star_shaped,
    find_kernel
)

class EllipseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maximum Inscribed Ellipse Finder")
        
        # Configure root window to be resizable
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize variables
        self.n_points = tk.IntVar(value=3)
        self.polygon = None
        self.inner_hull = None
        self.C = None
        self.d = None
        self.kernel_point = None
        
        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure main frame grid
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        # Create control panel
        self.create_control_panel()
        
        # Create plot area
        self.create_plot_area()
        
        # Initial generation
        self.generate_polygon()

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Number of points control
        ttk.Label(control_frame, text="Number of points:").grid(row=0, column=0, sticky=tk.W)
        self.points_spinbox = ttk.Spinbox(control_frame, from_=3, to=100, textvariable=self.n_points, width=5)
        self.points_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.update_btn = ttk.Button(control_frame, text="Update", command=self.generate_polygon)
        self.update_btn.grid(row=0, column=2, padx=5)

    def create_plot_area(self):
        # Create frame for the plot
        plot_frame = ttk.Frame(self.main_frame)
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Configure plot frame grid
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)
        
        # Create figure
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def generate_polygon(self):
        n_points = self.n_points.get()
        self.polygon = generate_random_convex_polygon_method(n_points)
        self.update_ellipse()

    def is_polygon_convex(self, polygon):
        """Check if a polygon is convex by checking if all cross products have the same sign."""
        if len(polygon) < 3:
            return False
        
        n = len(polygon)
        sign = None
        
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            p3 = polygon[(i + 2) % n]
            
            # Calculate cross product
            v1 = p2 - p1
            v2 = p3 - p2
            cross = np.cross(v1, v2)
            
            if abs(cross) > 1e-10:  # Ignore nearly zero cross products
                if sign is None:
                    sign = np.sign(cross)
                elif np.sign(cross) != sign:
                    return False
        
        return True

    def update_ellipse(self):
        if self.polygon is None:
            return
        
        self.inner_hull = None
        self.C = None
        self.d = None
        
        try:
            # Try star-shaped approach first
            self.kernel_point = find_kernel(self.polygon)
            if self.kernel_point is None:
                raise ValueError("The polygon is not star-shaped or no kernel point could be found")
            self.C, self.d = find_max_inscribed_ellipse_star(self.polygon)
                    
        except Exception as e:
            print(f"Error in star-shaped approach: {e}")
            # Fallback: try the inscribed circle approach
            try:
                print("Trying fallback inscribed circle approach...")
                self.inner_hull = find_inner_hull_via_inscribed_circle(self.polygon)
                self.C, self.d = find_max_inscribed_ellipse(self.inner_hull)
                print("Fallback successful")
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                return
                
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        # Plot outer polygon
        if self.polygon is not None:
            self.ax.plot(*self.polygon.T, 'b-', linewidth=2, label='Polygon', alpha=0.7)
            self.ax.plot([self.polygon[-1,0], self.polygon[0,0]], 
                        [self.polygon[-1,1], self.polygon[0,1]], 'b-', linewidth=2, alpha=0.7)
            self.ax.scatter(self.polygon[:, 0], self.polygon[:, 1], c='blue', s=30, alpha=0.7)
        
        # Plot inscribed ellipse
        if self.C is not None and self.d is not None:
            x, y = ellipse_points(self.C, self.d)
            self.ax.plot(x, y, 'r-', linewidth=3, label='Max Area Inscribed Ellipse')
            # Add ellipse info
            area = np.pi * np.linalg.det(self.C)
            self.ax.text(0.02, 0.98, f'Ellipse Area: {area:.4f}',
                       transform=self.ax.transAxes, verticalalignment='top')
        
        self.ax.axis('equal')
        self.ax.grid(True, alpha=0.3)
        if self.polygon is not None and self.C is not None:
            self.ax.legend()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = EllipseApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()