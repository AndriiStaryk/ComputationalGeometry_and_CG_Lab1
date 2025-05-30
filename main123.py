# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# # Перевірка залежностей
# try:
#     import tkinter as tk
#     from tkinter import ttk, messagebox, filedialog
# except ImportError:
#     print("Помилка: tkinter не встановлений.")
#     print("На macOS спробуйте: brew install python-tk")
#     exit(1)

# try:
#     import numpy as np
# except ImportError:
#     print("Помилка: numpy не встановлений.")
#     print("Встановіть: pip install numpy")
#     exit(1)

# try:
#     import matplotlib.pyplot as plt
#     from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#     from matplotlib.figure import Figure
# except ImportError:
#     print("Помилка: matplotlib не встановлений.")
#     print("Встановіть: pip install matplotlib")
#     exit(1)

# try:
#     import cvxpy as cp
# except ImportError:
#     print("Помилка: cvxpy не встановлений.")
#     print("Встановіть: pip install cvxpy")
#     exit(1)

# import re
# from shapely.geometry import Point, Polygon
# from shapely.ops import transform
# import warnings
# warnings.filterwarnings('ignore')


# class EllipseInscriber:
#     def __init__(self):
#         self.root = tk.Tk()
#         self.root.title("Вписування еліпса в зірковий многокутник")
#         self.root.geometry("1200x800")
        
#         self.points = []
#         self.ellipse_params = None
#         self.click_mode = False
        
#         # Налаштування для графіка
#         self.plot_xlim = [-10, 10]
#         self.plot_ylim = [-10, 10]
        
#         self.setup_ui()
        
#     def setup_ui(self):
#         # Головний контейнер
#         main_frame = ttk.Frame(self.root)
#         main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
#         # Ліва панель для введення даних
#         left_frame = ttk.Frame(main_frame, width=300)
#         left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
#         left_frame.pack_propagate(False)
        
#         # Заголовок
#         title_label = ttk.Label(left_frame, text="Вписування еліпса в многокутник", 
#                                font=("Arial", 14, "bold"))
#         title_label.pack(pady=(0, 20))
        
#         # Секція введення координат
#         coord_frame = ttk.LabelFrame(left_frame, text="Введення координат", padding=10)
#         coord_frame.pack(fill=tk.X, pady=(0, 10))
        
#         ttk.Label(coord_frame, text="Формат: (x1;y1), (x2;y2), ...").pack(anchor=tk.W)
        
#         self.coord_text = tk.Text(coord_frame, height=4, width=35)
#         self.coord_text.pack(fill=tk.X, pady=5)
        
#         coord_buttons_frame = ttk.Frame(coord_frame)
#         coord_buttons_frame.pack(fill=tk.X)
        
#         ttk.Button(coord_buttons_frame, text="Застосувати", 
#                   command=self.apply_coordinates).pack(side=tk.LEFT, padx=(0, 5))
#         ttk.Button(coord_buttons_frame, text="Очистити", 
#                   command=self.clear_coordinates).pack(side=tk.LEFT)
        
#         # Секція введення мишкою
#         mouse_frame = ttk.LabelFrame(left_frame, text="Введення мишкою", padding=10)
#         mouse_frame.pack(fill=tk.X, pady=(0, 10))
        
#         self.click_button = ttk.Button(mouse_frame, text="Почати введення точок", 
#                                       command=self.toggle_click_mode)
#         self.click_button.pack(fill=tk.X, pady=2)
        
#         ttk.Button(mouse_frame, text="Очистити точки", 
#                   command=self.clear_points).pack(fill=tk.X, pady=2)
        
#         # Секція налаштувань графіка
#         view_frame = ttk.LabelFrame(left_frame, text="Налаштування вигляду", padding=10)
#         view_frame.pack(fill=tk.X, pady=(0, 10))
        
#         ttk.Button(view_frame, text="Автомасштаб", 
#                   command=self.auto_scale).pack(fill=tk.X, pady=2)
#         ttk.Button(view_frame, text="Скинути масштаб", 
#                   command=self.reset_scale).pack(fill=tk.X, pady=2)
        
#         # Секція обчислень
#         calc_frame = ttk.LabelFrame(left_frame, text="Обчислення", padding=10)
#         calc_frame.pack(fill=tk.X, pady=(0, 10))
        
#         ttk.Button(calc_frame, text="Знайти еліпс", 
#                   command=self.find_ellipse).pack(fill=tk.X, pady=2)
#         ttk.Button(calc_frame, text="Очистити все", 
#                   command=self.clear_all).pack(fill=tk.X, pady=2)
        
#         # Секція збереження
#         save_frame = ttk.LabelFrame(left_frame, text="Збереження", padding=10)
#         save_frame.pack(fill=tk.X, pady=(0, 10))
        
#         ttk.Button(save_frame, text="Зберегти координати", 
#                   command=self.save_coordinates).pack(fill=tk.X, pady=2)
#         ttk.Button(save_frame, text="Завантажити координати", 
#                   command=self.load_coordinates).pack(fill=tk.X, pady=2)
#         ttk.Button(save_frame, text="Зберегти графік", 
#                   command=self.save_plot).pack(fill=tk.X, pady=2)
        
#         # Приклади для швидкого тестування
#         examples_frame = ttk.LabelFrame(left_frame, text="Приклади", padding=10)
#         examples_frame.pack(fill=tk.X, pady=(0, 10))
        
#         ttk.Button(examples_frame, text="Квадрат", 
#                   command=self.load_square_example).pack(fill=tk.X, pady=1)
#         ttk.Button(examples_frame, text="Зірка", 
#                   command=self.load_star_example).pack(fill=tk.X, pady=1)
#         ttk.Button(examples_frame, text="Шестикутник", 
#                   command=self.load_hexagon_example).pack(fill=tk.X, pady=1)
        
#         # Інформаційна панель
#         info_frame = ttk.LabelFrame(left_frame, text="Інформація", padding=10)
#         info_frame.pack(fill=tk.BOTH, expand=True)
        
#         self.info_text = tk.Text(info_frame, height=6, width=35, state=tk.DISABLED)
#         self.info_text.pack(fill=tk.BOTH, expand=True)
        
#         # Права панель для графіка
#         right_frame = ttk.Frame(main_frame)
#         right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
#         # Створення matplotlib фігури
#         self.fig = Figure(figsize=(10, 8), dpi=100)
#         self.ax = self.fig.add_subplot(111)
#         self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
#         self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
#         # Прив'язка події кліку
#         self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
#         self.update_plot()
        
#     def toggle_click_mode(self):
#         self.click_mode = not self.click_mode
#         if self.click_mode:
#             self.click_button.config(text="Зупинити введення точок")
#             self.update_info("Натискайте на графіку для додавання точок")
#         else:
#             self.click_button.config(text="Почати введення точок")
#             self.update_info("Режим введення точок вимкнено")
    
#     def on_canvas_click(self, event):
#         if self.click_mode and event.inaxes:
#             x, y = event.xdata, event.ydata
#             self.points.append((x, y))
#             self.update_coordinates_text()
#             self.update_info(f"Додано точку: ({x:.2f}, {y:.2f})\nВсього точок: {len(self.points)}")
#             # Зберігаємо поточні межі графіка
#             current_xlim = self.ax.get_xlim()
#             current_ylim = self.ax.get_ylim()
#             self.update_plot(keep_limits=True)
#             # Відновлюємо межі після оновлення
#             self.ax.set_xlim(current_xlim)
#             self.ax.set_ylim(current_ylim)
#             self.canvas.draw()
    
#     def apply_coordinates(self):
#         text = self.coord_text.get("1.0", tk.END).strip()
#         if not text:
#             messagebox.showwarning("Попередження", "Введіть координати!")
#             return
        
#         try:
#             points = []
            
#             # Видаляємо зайві символи та розділяємо по комах
#             text = re.sub(r'[^\d\-.,;()\s]', '', text)
            
#             # Шукаємо пари чисел у дужках або без них
#             patterns = [
#                 r'\(\s*([+-]?\d*\.?\d+)\s*[;,]\s*([+-]?\d*\.?\d+)\s*\)',  # (x;y) або (x,y)
#                 r'([+-]?\d*\.?\d+)\s*[;,]\s*([+-]?\d*\.?\d+)'  # x;y або x,y
#             ]
            
#             for pattern in patterns:
#                 matches = re.findall(pattern, text)
#                 if matches:
#                     points = [(float(x), float(y)) for x, y in matches]
#                     break
            
#             if not points:
#                 raise ValueError("Не вдалося розпізнати координати")
            
#             if len(points) < 3:
#                 raise ValueError("Потрібно мінімум 3 точки")
            
#             self.points = points
#             self.ellipse_params = None
#             self.auto_scale()
#             self.update_info(f"Завантажено {len(points)} точок")
            
#         except Exception as e:
#             messagebox.showerror("Помилка", f"Помилка при обробці координат: {str(e)}")
    
#     def update_coordinates_text(self):
#         coord_str = ", ".join([f"({x:.2f};{y:.2f})" for x, y in self.points])
#         self.coord_text.delete("1.0", tk.END)
#         self.coord_text.insert("1.0", coord_str)
    
#     def clear_coordinates(self):
#         self.coord_text.delete("1.0", tk.END)
    
#     def clear_points(self):
#         self.points = []
#         self.ellipse_params = None
#         self.update_plot()
#         self.clear_coordinates()
#         self.update_info("Точки очищено")
    
#     def clear_all(self):
#         self.clear_points()
#         self.reset_scale()
#         self.update_info("Все очищено")
    
#     def auto_scale(self):
#         """Автоматично встановлює масштаб графіка під точки"""
#         if not self.points:
#             return
            
#         points = np.array(self.points)
#         x_min, x_max = points[:, 0].min(), points[:, 0].max()
#         y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
#         # Додаємо відступи
#         x_margin = (x_max - x_min) * 0.2 if x_max != x_min else 1
#         y_margin = (y_max - y_min) * 0.2 if y_max != y_min else 1
        
#         self.plot_xlim = [x_min - x_margin, x_max + x_margin]
#         self.plot_ylim = [y_min - y_margin, y_max + y_margin]
        
#         self.update_plot()
    
#     def reset_scale(self):
#         """Скидає масштаб до значень за замовчуванням"""
#         self.plot_xlim = [-10, 10]
#         self.plot_ylim = [-10, 10]
#         self.update_plot()
    
#     def load_square_example(self):
#         """Завантажує приклад квадрата"""
#         self.points = [(0, 0), (4, 0), (4, 4), (0, 4)]
#         self.ellipse_params = None
#         self.update_coordinates_text()
#         self.auto_scale()
#         self.update_info("Завантажено приклад: квадрат")
    
#     def load_star_example(self):
#         """Завантажує приклад зірки"""
#         # Створюємо 5-кутну зірку
#         points = []
#         for i in range(10):
#             angle = i * np.pi / 5
#             radius = 3 if i % 2 == 0 else 1.5
#             x = radius * np.cos(angle)
#             y = radius * np.sin(angle)
#             points.append((x, y))
        
#         self.points = points
#         self.ellipse_params = None
#         self.update_coordinates_text()
#         self.auto_scale()
#         self.update_info("Завантажено приклад: зірка")
    
#     def load_hexagon_example(self):
#         """Завантажує приклад шестикутника"""
#         points = []
#         for i in range(6):
#             angle = i * np.pi / 3
#             x = 3 * np.cos(angle)
#             y = 3 * np.sin(angle)
#             points.append((x, y))
        
#         self.points = points
#         self.ellipse_params = None
#         self.update_coordinates_text()
#         self.auto_scale()
#         self.update_info("Завантажено приклад: шестикутник")
    
#     def find_ellipse(self):
#         if len(self.points) < 3:
#             messagebox.showwarning("Попередження", "Потрібно мінімум 3 точки!")
#             return
        
#         try:
#             self.update_info("Обчислення еліпса...")
#             self.root.update()  # Оновлюємо інтерфейс
            
#             # Використовуємо більш надійний алгоритм
#             self.ellipse_params = self.inscribe_ellipse_robust(self.points)
            
#             if self.ellipse_params is None:
#                 self.update_info("Не вдалося знайти еліпс. Спробуйте інші точки.")
#                 return
            
#             self.update_plot()
            
#             # Виводимо інформацію про еліпс
#             h, k, a, b, theta = self.ellipse_params
#             area = np.pi * abs(a) * abs(b)
            
#             info = f"ЕЛІПС ЗНАЙДЕНО!\n"
#             info += f"Центр: ({h:.3f}, {k:.3f})\n"
#             info += f"Півосі: a={abs(a):.3f}, b={abs(b):.3f}\n"
#             info += f"Кут повороту: {np.degrees(theta):.1f}°\n"
#             info += f"Площа: {area:.3f}\n"
#             info += f"Точок многокутника: {len(self.points)}"
            
#             self.update_info(info)
            
#         except Exception as e:
#             messagebox.showerror("Помилка", f"Помилка при обчисленні еліпса: {str(e)}")
#             self.update_info(f"Помилка: {str(e)}")
    
#     def point_in_polygon(self, point, polygon_points):
#         """Перевіряє, чи точка знаходиться всередині многокутника (ray casting algorithm)"""
#         x, y = point
#         n = len(polygon_points)
#         inside = False
        
#         p1x, p1y = polygon_points[0]
#         for i in range(1, n + 1):
#             p2x, p2y = polygon_points[i % n]
#             if y > min(p1y, p2y):
#                 if y <= max(p1y, p2y):
#                     if x <= max(p1x, p2x):
#                         if p1y != p2y:
#                             xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
#                         if p1x == p2x or x <= xinters:
#                             inside = not inside
#             p1x, p1y = p2x, p2y
        
#         return inside
    
#     def ellipse_polygon_intersection(self, ellipse_params, polygon_points, num_samples=100):
#         """Перевіряє, чи еліпс повністю міститься всередині многокутника"""
#         h, k, a, b, theta = ellipse_params
        
#         # Генеруємо точки на контурі еліпса
#         t = np.linspace(0, 2*np.pi, num_samples)
#         x_ellipse = a * np.cos(t)
#         y_ellipse = b * np.sin(t)
        
#         # Повертаємо еліпс
#         cos_theta = np.cos(theta)
#         sin_theta = np.sin(theta)
#         x_rot = x_ellipse * cos_theta - y_ellipse * sin_theta + h
#         y_rot = x_ellipse * sin_theta + y_ellipse * cos_theta + k
        
#         # Перевіряємо, чи всі точки еліпса всередині многокутника
#         for i in range(num_samples):
#             if not self.point_in_polygon((x_rot[i], y_rot[i]), polygon_points):
#                 return False
        
#         return True
    
#     def distance_point_to_polygon_edge(self, point, polygon_points):
#         """Обчислює мінімальну відстань від точки до найближчого ребра многокутника"""
#         min_distance = float('inf')
#         x0, y0 = point
        
#         n = len(polygon_points)
#         for i in range(n):
#             # Ребро від точки i до точки (i+1) % n
#             x1, y1 = polygon_points[i]
#             x2, y2 = polygon_points[(i + 1) % n]
            
#             # Обчислюємо відстань від точки до відрізка
#             A = x0 - x1
#             B = y0 - y1
#             C = x2 - x1
#             D = y2 - y1
            
#             dot = A * C + B * D
#             len_sq = C * C + D * D
            
#             if len_sq == 0:  # Точки співпадають
#                 distance = np.sqrt(A * A + B * B)
#             else:
#                 param = dot / len_sq
#                 if param < 0:
#                     xx, yy = x1, y1
#                 elif param > 1:
#                     xx, yy = x2, y2
#                 else:
#                     xx = x1 + param * C
#                     yy = y1 + param * D
                
#                 dx = x0 - xx
#                 dy = y0 - yy
#                 distance = np.sqrt(dx * dx + dy * dy)
            
#             min_distance = min(min_distance, distance)
        
#         return min_distance
    
#     def polygon_to_Ab(self, points):
#         # points: Nx2 array, ordered counterclockwise
#         n = len(points)
#         A = []
#         b = []
#         for i in range(n):
#             p1 = np.array(points[i])
#             p2 = np.array(points[(i+1)%n])
#             edge = p2 - p1
#             normal = np.array([edge[1], -edge[0]])  # outward normal
#             normal = normal / np.linalg.norm(normal)
#             A.append(normal)
#             b.append(np.dot(normal, p1))
#         return np.array(A), np.array(b)
    
#     def inscribe_ellipse_robust(self, points):
#         """Знаходить максимальний вписаний еліпс використовуючи H-представлення (Ax <= b)"""
#         if len(points) < 3:
#             return None
        
#         pts = np.array(points)
#         n = 2
#         # Використовуємо вершини многокутника для H-представлення
#         A, b = self.polygon_to_Ab(pts)
#         Bvar = cp.Variable((n, n), symmetric=True)
#         dvar = cp.Variable(n)
#         constraints = []
#         for i in range(len(A)):
#             constraints.append(cp.norm(Bvar @ A[i]) + A[i] @ dvar <= b[i])
#         objective = cp.Minimize(-cp.log_det(Bvar))
#         problem = cp.Problem(objective, constraints)
#         try:
#             problem.solve(solver=cp.SCS)
#             if problem.status == 'optimal':
#                 B = Bvar.value
#                 d = dvar.value
#                 evals, evecs = np.linalg.eigh(B)
#                 a = 1/np.sqrt(evals[1])
#                 b_ = 1/np.sqrt(evals[0])
#                 theta = np.arctan2(evecs[1,1], evecs[0,1])
#                 h, k = d
#                 return (h, k, a, b_, theta)
#             else:
#                 return None
#         except Exception as e:
#             print(f'Optimization error: {e}')
#             return None
    
#     def update_plot(self, keep_limits=False):
#         self.ax.clear()
        
#         if self.points:
#             # Малюємо многокутник
#             points_array = np.array(self.points + [self.points[0]])  # Замикаємо многокутник
#             self.ax.plot(points_array[:, 0], points_array[:, 1], 'b-', linewidth=2, label='Многокутник')
#             self.ax.scatter([p[0] for p in self.points], [p[1] for p in self.points], 
#                            c='blue', s=60, zorder=5, edgecolor='white', linewidth=1)
            
#             # Нумеруємо точки
#             for i, (x, y) in enumerate(self.points):
#                 self.ax.annotate(f'{i+1}', (x, y), xytext=(8, 8), 
#                                textcoords='offset points', fontsize=9, 
#                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
#         if self.ellipse_params is not None:
#             # Малюємо еліпс
#             h, k, a, b, theta = self.ellipse_params
            
#             # Перевіряємо коректність параметрів
#             if a > 0 and b > 0:
#                 # Створюємо точки еліпса
#                 t = np.linspace(0, 2*np.pi, 200)
#                 x_ellipse = a * np.cos(t)
#                 y_ellipse = b * np.sin(t)
                
#                 # Повертаємо еліпс
#                 cos_theta = np.cos(theta)
#                 sin_theta = np.sin(theta)
#                 x_rot = x_ellipse * cos_theta - y_ellipse * sin_theta + h
#                 y_rot = x_ellipse * sin_theta + y_ellipse * cos_theta + k
                
#                 self.ax.plot(x_rot, y_rot, 'r-', linewidth=3, label='Вписаний еліпс', alpha=0.8)
#                 self.ax.scatter([h], [k], c='red', s=100, marker='x', zorder=10, 
#                                linewidth=3, label='Центр еліпса')
                
#                 # Малюємо осі еліпса
#                 # Велику вісь
#                 major_x = [h - a*cos_theta, h + a*cos_theta]
#                 major_y = [k - a*sin_theta, k + a*sin_theta]
#                 self.ax.plot(major_x, major_y, 'r--', alpha=0.5, linewidth=1)
                
#                 # Малу вісь
#                 minor_x = [h + b*sin_theta, h - b*sin_theta]
#                 minor_y = [k - b*cos_theta, k + b*cos_theta]
#                 self.ax.plot(minor_x, minor_y, 'r--', alpha=0.5, linewidth=1)
        
#         # Встановлюємо межі графіка
#         if not keep_limits:
#             self.ax.set_xlim(self.plot_xlim)
#             self.ax.set_ylim(self.plot_ylim)
        
#         self.ax.set_aspect('equal')
#         self.ax.grid(True, alpha=0.3)
#         self.ax.set_xlabel('X', fontsize=12)
#         self.ax.set_ylabel('Y', fontsize=12)
#         self.ax.set_title('Вписування еліпса в зірковий многокутник', fontsize=14)
        
#         if self.points or self.ellipse_params is not None:
#             self.ax.legend(loc='upper right')
        
#         self.fig.tight_layout()
#         self.canvas.draw()
    
#     def update_info(self, text):
#         self.info_text.config(state=tk.NORMAL)
#         self.info_text.delete("1.0", tk.END)
#         self.info_text.insert("1.0", text)
#         self.info_text.config(state=tk.DISABLED)
    
#     def save_coordinates(self):
#         if not self.points:
#             messagebox.showwarning("Попередження", "Немає точок для збереження!")
#             return
        
#         filename = filedialog.asksaveasfilename(
#             defaultextension=".txt",
#             filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
#         )
        
#         if filename:
#             try:
#                 with open(filename, 'w', encoding='utf-8') as f:
#                     for x, y in self.points:
#                         f.write(f"({x:.6f};{y:.6f})\n")
#                 self.update_info(f"Координати збережено в {filename}")
#             except Exception as e:
#                 messagebox.showerror("Помилка", f"Помилка збереження: {str(e)}")
    
#     def load_coordinates(self):
#         filename = filedialog.askopenfilename(
#             filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
#         )
        
#         if filename:
#             try:
#                 with open(filename, 'r', encoding='utf-8') as f:
#                     content = f.read()
                
#                 self.coord_text.delete("1.0", tk.END)
#                 self.coord_text.insert("1.0", content)
#                 self.apply_coordinates()
                
#             except Exception as e:
#                 messagebox.showerror("Помилка", f"Помилка завантаження: {str(e)}")
    
#     def save_plot(self):
#         if not self.points:
#             messagebox.showwarning("Попередження", "Немає даних для збереження!")
#             return
        
#         filename = filedialog.asksaveasfilename(
#             defaultextension=".png",
#             filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
#         )
        
#         if filename:
#             try:
#                 self.fig.savefig(filename, dpi=300, bbox_inches='tight')
#                 self.update_info(f"Графік збережено в {filename}")
#             except Exception as e:
#                 messagebox.showerror("Помилка", f"Помилка збереження: {str(e)}")
    
#     def run(self):
#         self.root.mainloop()


# if __name__ == "__main__":
#     app = EllipseInscriber()
#     app.run()