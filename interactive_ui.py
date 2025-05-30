import matplotlib.pyplot as plt
import numpy as np
from mve import mve
from ui import plot_ellipse
from scipy.spatial import ConvexHull

class InteractiveEllipsoidFinder:
    def __init__(self):
        self.points = []
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_title("Click to add points. Press 'Enter' to compute ellipsoid.\nPress 'c' to clear points.")
        
        # Set fixed scale
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Add instructions text
        self.instructions = self.ax.text(0.02, 0.98, 
            "Instructions:\n1. Click to add points\n2. Press Enter to compute ellipsoid\n3. Press 'c' to clear points",
            transform=self.ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Connect the events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Show the plot
        plt.show()
    
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        # Add the clicked point
        self.points.append([event.xdata, event.ydata])
        
        # Plot the point
        self.ax.plot(event.xdata, event.ydata, 'bo', markersize=8)
        
        # Draw a small cross at the point for better visibility
        self.ax.plot(event.xdata, event.ydata, 'r+', markersize=8)
        
        # Update the plot
        self.fig.canvas.draw()
    
    def on_key(self, event):
        if event.key == 'enter' and len(self.points) >= 3:
            points = np.array(self.points)
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                # Draw the polygon
                self.ax.plot(np.append(hull_points[:,0], hull_points[0,0]),
                             np.append(hull_points[:,1], hull_points[0,1]), 'g-', lw=2)
                # Compute centroid
                centroid = np.mean(hull_points, axis=0)
                self.ax.plot(centroid[0], centroid[1], 'ko', markersize=6, label='Centroid')
                # Build Ax <= b for each edge
                A = []
                b = []
                for i in range(len(hull_points)):
                    p1 = hull_points[i]
                    p2 = hull_points[(i+1)%len(hull_points)]
                    edge = p2 - p1
                    # Outward normal (robust)
                    normal = np.array([edge[1], -edge[0]])
                    normal = normal / np.linalg.norm(normal)
                    # Check direction: normal should point away from centroid
                    midpoint = (p1 + p2) / 2
                    to_centroid = centroid - midpoint
                    if np.dot(normal, to_centroid) > 0:
                        normal = -normal
                    A.append(normal)
                    b.append(np.dot(normal, p1))
                    # Plot normal for debug
                    self.ax.arrow(midpoint[0], midpoint[1], normal[0], normal[1],
                                  head_width=0.3, head_length=0.5, fc='m', ec='m')
                A = np.array(A)
                b = np.array(b)
                # Compute the maximum volume ellipsoid
                B, d = mve(A, b)
                from get_axes import get_axes
                axes = get_axes(B, d)
                # Use SVD of B to get the correct orientation
                U, D, Vt = np.linalg.svd(B)
                angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
                
                # Create ellipse patch
                ellipse = plt.matplotlib.patches.Ellipse(
                    xy=d, 
                    width=2*axes[0], 
                    height=2*axes[1], 
                    angle=angle,
                    edgecolor='r', 
                    fc='None', 
                    lw=2
                )
                self.ax.add_patch(ellipse)
                
                # Update the plot
                self.fig.canvas.draw()
                
            except Exception as e:
                print(f"Error computing ellipsoid: {e}")
        
        elif event.key == 'c':
            # Clear all points and ellipses
            self.points = []
            self.ax.clear()
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
            self.ax.grid(True)
            self.ax.set_aspect('equal')
            self.ax.set_title("Click to add points. Press 'Enter' to compute ellipsoid.\nPress 'c' to clear points.")
            self.instructions = self.ax.text(0.02, 0.98, 
                "Instructions:\n1. Click to add points\n2. Press Enter to compute ellipsoid\n3. Press 'c' to clear points",
                transform=self.ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            self.fig.canvas.draw()

if __name__ == "__main__":
    InteractiveEllipsoidFinder() 