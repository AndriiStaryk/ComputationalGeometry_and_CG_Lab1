import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from max_ellipse import (
    generate_random_convex_polygon_method1,
    generate_random_convex_polygon_method2,
    generate_random_convex_polygon_method3,
    generate_random_convex_polygon_method4,
    find_inner_convex_hull,
    find_inner_hull_via_inscribed_circle,
    find_inner_hull_via_shrinking,
    find_max_inscribed_ellipse,
    ellipse_points
)

class EllipseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maximum Inscribed Ellipse Finder")
        
        # Configure root window to be resizable
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize variables
        self.n_points = tk.IntVar(value=10)
        self.polygon = None
        self.inner_hull = None
        self.C = None
        self.d = None
        
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
        points_spinbox = ttk.Spinbox(control_frame, from_=3, to=100, textvariable=self.n_points, width=5)
        points_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(control_frame, text="Update", command=self.generate_polygon).grid(row=0, column=2, padx=5)
        
        # Polygon generation methods
        ttk.Label(control_frame, text="Polygon Generation:").grid(row=1, column=0, sticky=tk.W, pady=(10,0))
        self.polygon_method = tk.StringVar(value="1")
        methods = [
            ("Random radii on circle", "1"),
            ("Points in annulus", "2"),
            ("Selection from candidates", "3"),
            ("Semi-regular with variation", "4")
        ]
        for i, (text, value) in enumerate(methods):
            ttk.Radiobutton(control_frame, text=text, value=value, 
                           variable=self.polygon_method,
                           command=self.generate_polygon).grid(row=2+i, column=0, columnspan=3, sticky=tk.W)
        
        # Inner hull methods
        ttk.Label(control_frame, text="Inner Hull Method:").grid(row=6, column=0, sticky=tk.W, pady=(10,0))
        self.inner_method = tk.StringVar(value="2")
        inner_methods = [
            ("Random points inside polygon", "1"),
            ("Inscribed circle approach", "2"),
            ("Shrinking approach", "3")
        ]
        for i, (text, value) in enumerate(inner_methods):
            ttk.Radiobutton(control_frame, text=text, value=value,
                           variable=self.inner_method,
                           command=self.update_inner_hull).grid(row=7+i, column=0, columnspan=3, sticky=tk.W)

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
        method = self.polygon_method.get()
        
        if method == "1":
            self.polygon = generate_random_convex_polygon_method1(n_points)
        elif method == "2":
            self.polygon = generate_random_convex_polygon_method2(n_points)
        elif method == "3":
            self.polygon = generate_random_convex_polygon_method3(n_points)
        elif method == "4":
            self.polygon = generate_random_convex_polygon_method4(n_points)
        
        self.update_inner_hull()

    def update_inner_hull(self):
        if self.polygon is None:
            return
            
        method = self.inner_method.get()
        
        if method == "1":
            self.inner_hull = find_inner_convex_hull(self.polygon, num_candidate_points=2000)
        elif method == "2":
            self.inner_hull = find_inner_hull_via_inscribed_circle(self.polygon)
        elif method == "3":
            self.inner_hull = find_inner_hull_via_shrinking(self.polygon)
        
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        
        # Plot outer polygon
        if self.polygon is not None:
            self.ax.plot(*self.polygon.T, 'b-', linewidth=2, label='Starry Polygon', alpha=0.7)
            self.ax.plot([self.polygon[-1,0], self.polygon[0,0]], 
                        [self.polygon[-1,1], self.polygon[0,1]], 'b-', linewidth=2, alpha=0.7)
            self.ax.scatter(self.polygon[:, 0], self.polygon[:, 1], c='blue', s=30, alpha=0.7)
        
        # Plot inner hull
        if self.inner_hull is not None:
            self.ax.plot(*self.inner_hull.T, 'g-', linewidth=2, label='Inner Convex Hull')
            self.ax.plot([self.inner_hull[-1,0], self.inner_hull[0,0]], 
                        [self.inner_hull[-1,1], self.inner_hull[0,1]], 'g-', linewidth=2)
            self.ax.scatter(self.inner_hull[:, 0], self.inner_hull[:, 1], c='green', s=50)
            
            # Find and plot inscribed ellipse
            try:
                self.C, self.d = find_max_inscribed_ellipse(self.inner_hull)
                x, y = ellipse_points(self.C, self.d)
                self.ax.plot(x, y, 'r-', linewidth=3, label='Max Area Inscribed Ellipse')
                
                # Add ellipse info
                area = np.pi * np.linalg.det(self.C)
                self.ax.text(0.02, 0.98, f'Ellipse Area: {area:.4f}',
                           transform=self.ax.transAxes, verticalalignment='top')
            except Exception as e:
                print(f"Error finding inscribed ellipse: {e}")
        
        self.ax.axis('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = EllipseApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 