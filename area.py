"""Interactive Discrete Disk Area Visualization Tool."""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
import numpy as np
from discrete_disk import DiscreteDisk, create_area_by_join, Coordinate, MODE_I, MODE_B, MODE_O, MODES, set_disk_mode


class AreaVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Discrete Disk Area Visualizer")
        self.root.geometry("1200x800")
        
        # Set protocol for window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize variables
        self.unit = 3
        self.disk_mode = 'sq_border'  # sq_center, sq_border, hex_center
        self.current_area = None
        self.disk_list = []
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        self.setup_ui()
        self.update_visualization()
        
        # Set initial discrete disk mode
        set_disk_mode(self.disk_mode)
    
    def on_closing(self):
        """Handle window closing event."""
        try:
            # Close matplotlib figure properly
            plt.close(self.fig)
        except:
            pass
        
        # Destroy the root window
        self.root.quit()
        self.root.destroy()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for visualization
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel for controls
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        self.setup_visualization_panel(left_frame)
        self.setup_control_panel(right_frame)
    
    def setup_visualization_panel(self, parent):
        """Setup the matplotlib visualization panel."""
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add click event handler
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
        # Setup axes
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Discrete Disk Area Visualization\n(Click to add disk at position)')
    
    def setup_control_panel(self, parent):
        """Setup the control panel."""
        # Title
        title_label = ttk.Label(parent, text="Control Panel", font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Unit setting
        unit_frame = ttk.LabelFrame(parent, text="Unit Radius", padding=10)
        unit_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.unit_var = tk.IntVar(value=self.unit)
        unit_scale = ttk.Scale(unit_frame, from_=1, to=10, variable=self.unit_var, 
                              orient=tk.HORIZONTAL, command=self.on_unit_changed)
        unit_scale.pack(fill=tk.X)
        
        unit_label = ttk.Label(unit_frame, textvariable=self.unit_var)
        unit_label.pack()
        
        # Disk type selection
        type_frame = ttk.LabelFrame(parent, text="Disk Type", padding=10)
        type_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.disk_type_var = tk.StringVar(value="connected")
        ttk.Radiobutton(type_frame, text="Connected", variable=self.disk_type_var, 
                       value="connected").pack(anchor=tk.W)
        ttk.Radiobutton(type_frame, text="Disconnected", variable=self.disk_type_var, 
                       value="disconnected").pack(anchor=tk.W)
        
        # Discrete disk mode selection
        mode_frame = ttk.LabelFrame(parent, text="Discrete Disk Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.disk_mode_var = tk.StringVar(value=self.disk_mode)
        ttk.Radiobutton(mode_frame, text="Square Border", variable=self.disk_mode_var,
                       value="sq_border", command=self.on_mode_changed).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Square Center", variable=self.disk_mode_var,
                       value="sq_center", command=self.on_mode_changed).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Hexagonal Center", variable=self.disk_mode_var,
                       value="hex_center", command=self.on_mode_changed).pack(anchor=tk.W)
        
        # Manual position entry
        pos_frame = ttk.LabelFrame(parent, text="Add Disk at Position", padding=10)
        pos_frame.pack(fill=tk.X, pady=(0, 10))
        
        pos_entry_frame = ttk.Frame(pos_frame)
        pos_entry_frame.pack(fill=tk.X)
        
        ttk.Label(pos_entry_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.x_var = tk.IntVar()
        ttk.Entry(pos_entry_frame, textvariable=self.x_var, width=8).grid(row=0, column=1, padx=(5, 10))
        
        ttk.Label(pos_entry_frame, text="Y:").grid(row=0, column=2, sticky=tk.W)
        self.y_var = tk.IntVar()
        ttk.Entry(pos_entry_frame, textvariable=self.y_var, width=8).grid(row=0, column=3, padx=5)
        
        ttk.Button(pos_frame, text="Add Disk", command=self.add_disk_manual).pack(pady=(10, 0))
        
        # Action buttons
        action_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(action_frame, text="Clear All", command=self.clear_all).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(action_frame, text="Remove Last", command=self.remove_last).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(action_frame, text="Show Points", command=self.show_points).pack(fill=tk.X)
        
        # Disk list
        list_frame = ttk.LabelFrame(parent, text="Current Disks", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for disk list
        self.disk_tree = ttk.Treeview(list_frame, columns=('X', 'Y', 'Unit', 'Type'), show='headings', height=8)
        self.disk_tree.heading('#1', text='X')
        self.disk_tree.heading('#2', text='Y')
        self.disk_tree.heading('#3', text='Unit')
        self.disk_tree.heading('#4', text='Type')
        self.disk_tree.column('#1', width=40)
        self.disk_tree.column('#2', width=40)
        self.disk_tree.column('#3', width=40)
        self.disk_tree.column('#4', width=80)
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.disk_tree.yview)
        self.disk_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.disk_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Area info
        info_frame = ttk.LabelFrame(parent, text="Area Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.info_label = ttk.Label(info_frame, text="No disks added", wraplength=250)
        self.info_label.pack()
    
    def on_unit_changed(self, value):
        """Handle unit radius change."""
        # Round to nearest integer to ensure discrete values
        rounded_value = round(float(value))
        self.unit = rounded_value
        # Update the variable to reflect the rounded value
        self.unit_var.set(rounded_value)
        self.recalculate_area()
        self.update_visualization()
    
    def on_mode_changed(self):
        """Handle discrete disk mode change."""
        new_mode = self.disk_mode_var.get()
        if new_mode != self.disk_mode:
            self.disk_mode = new_mode
            set_disk_mode(self.disk_mode)
            self.recalculate_area()
            self.update_visualization()
    
    def on_canvas_click(self, event):
        """Handle canvas click to add disk."""
        if event.inaxes != self.ax:
            return
        
        # Convert real coordinates back to discrete coordinates
        if self.disk_mode == 'hex_center':
            from discrete_disk import SQRT3
            # For hex_center mode: real_x = x * SQRT3/2, real_y = y/2
            # So: x = real_x * 2/SQRT3, y = real_y * 2
            discrete_x = int(round(event.xdata * 2 / SQRT3))
            discrete_y = int(round(event.ydata * 2))
        else:
            discrete_x = int(round(event.xdata))
            discrete_y = int(round(event.ydata))
            
        self.add_disk(discrete_x, discrete_y, self.disk_type_var.get() == "connected")
    
    def add_disk_manual(self):
        """Add disk using manual position entry."""
        try:
            x = self.x_var.get()
            y = self.y_var.get()
            self.add_disk(x, y, self.disk_type_var.get() == "connected")
        except tk.TclError:
            messagebox.showerror("Error", "Please enter valid integer coordinates")
    
    def add_disk(self, x, y, connected):
        """Add a new disk at the specified position."""
        disk_info = {
            'x': x,
            'y': y,
            'unit': self.unit,  # Remember the size of each disk
            'connected': connected,
            'disk': DiscreteDisk.disk(self.unit, x, y, connected = 1 if connected else 0)
        }
        
        self.disk_list.append(disk_info)
        self.recalculate_area()
        self.update_disk_list()
        self.update_visualization()
        self.update_info()
    
    def recalculate_area(self):
        """Recalculate the joined area from all disks."""
        if not self.disk_list:
            self.current_area = None
            return
        
        # Start with the first disk using its remembered size
        first_disk = self.disk_list[0]
        self.current_area = DiscreteDisk.disk(first_disk['unit'], first_disk['x'], 
                                            first_disk['y'], 
                                            connected=1 if first_disk['connected'] else 0)
        
        # Join with subsequent disks using their remembered sizes
        for disk_info in self.disk_list[1:]:
            disk = DiscreteDisk.disk(disk_info['unit'], disk_info['x'], disk_info['y'], 
                                   connected=1 if disk_info['connected'] else 0)
            self.current_area = create_area_by_join(self.current_area, disk)
    
    def update_disk_list(self):
        """Update the disk list display."""
        # Clear existing items
        for item in self.disk_tree.get_children():
            self.disk_tree.delete(item)
        
        # Add current disks
        for i, disk_info in enumerate(self.disk_list):
            disk_type = "Connected" if disk_info['connected'] else "Disconnected"
            unit = disk_info.get('unit', self.unit)  # Fallback to current unit if not stored
            self.disk_tree.insert('', 'end', values=(disk_info['x'], disk_info['y'], unit, disk_type))
    
    def update_visualization(self):
        """Update the matplotlib visualization."""
        self.ax.clear()
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Discrete Disk Area Visualization\n(Click to add disk at position)')
        
        # Draw individual disks
        for i, disk_info in enumerate(self.disk_list):
            color = self.colors[i % len(self.colors)]
            alpha = 0.3
            linestyle = '-' if disk_info['connected'] else '--'
            
            # Use real coordinates for display
            real_x = disk_info['x']
            real_y = disk_info['y']
            disk_unit = disk_info.get('unit', self.unit)  # Use stored unit size
            real_radius = disk_unit
            
            # Apply scaling for hex_center mode
            if self.disk_mode == 'hex_center':
                from discrete_disk import SQRT3
                real_x = disk_info['x'] * SQRT3 / 2
                real_y = disk_info['y'] / 2
            
            circle = Circle((real_x, real_y), real_radius, 
                          color=color, alpha=alpha, linestyle=linestyle, fill=False, linewidth=2)
            self.ax.add_patch(circle)
            
            # Add label at real coordinates
            self.ax.text(real_x, real_y, str(i+1), 
                        ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw result area points if available
        if self.current_area:
            self.draw_area_points()
        
        self.canvas.draw()
    
    def draw_area_points(self):
        """Draw the points of the current area."""
        if not self.current_area:
            return
        
        # Get points from the area - call points_list for each type separately
        points_I = []
        points_B = []
        
        # Get Interior points
        try:
            for point in self.current_area.points_list(types='I'):
                real_x = point.get_real_x()
                real_y = point.get_real_y()
                points_I.append((real_x, real_y))
        except:
            pass
            
        # Get Boundary points  
        try:
            for point in self.current_area.points_list(types='B'):
                real_x = point.get_real_x()
                real_y = point.get_real_y()
                points_B.append((real_x, real_y))
        except:
            pass
        
        # Plot points with different colors based on type
        if points_I:
            x_coords, y_coords = zip(*points_I)
            self.ax.scatter(x_coords, y_coords, c='red', s=20, alpha=0.8, marker='s', label='Interior (I)')
        
        if points_B:
            x_coords, y_coords = zip(*points_B)
            self.ax.scatter(x_coords, y_coords, c='orange', s=15, alpha=0.6, marker='o', label='Boundary (B)')
        
        # Add legend if there are points
        if points_I or points_B:
            self.ax.legend(loc='upper right')
    
    def update_info(self):
        """Update the area information display."""
        if not self.current_area:
            self.info_label.config(text="No disks added")
            return
        
        # Count points by type
        count_I = 0
        count_B = 0
        
        try:
            count_I = sum(1 for p in self.current_area.points_list(types='I'))
        except:
            pass
            
        try:
            count_B = sum(1 for p in self.current_area.points_list(types='B'))
        except:
            pass
        
        info_text = f"Disks: {len(self.disk_list)}\n"
        info_text += f"Unit radius: {self.unit}\n"
        info_text += f"Mode: {self.disk_mode}\n"
        info_text += f"Interior points: {count_I}\n"
        info_text += f"Boundary points: {count_B}\n"
        info_text += f"Total points: {count_I + count_B}"
        
        self.info_label.config(text=info_text)
    
    def clear_all(self):
        """Clear all disks."""
        self.disk_list.clear()
        self.current_area = None
        self.update_disk_list()
        self.update_visualization()
        self.update_info()
    
    def remove_last(self):
        """Remove the last added disk."""
        if self.disk_list:
            self.disk_list.pop()
            self.recalculate_area()
            self.update_disk_list()
            self.update_visualization()
            self.update_info()
    
    def show_points(self):
        """Show detailed point information in a new window."""
        if not self.current_area:
            messagebox.showinfo("Info", "No area to display")
            return
        
        # Create new window
        points_window = tk.Toplevel(self.root)
        points_window.title("Area Points Details")
        points_window.geometry("400x500")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(points_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add point information
        text_widget.insert(tk.END, f"Area Points (Unit: {self.unit})\n")
        text_widget.insert(tk.END, "=" * 30 + "\n\n")
        
        # Get Interior points
        try:
            points_I = list(self.current_area.points_list(types='I'))
            if points_I:
                text_widget.insert(tk.END, f"Interior Points ({len(points_I)}): \n")
                for point in sorted(points_I, key=lambda p: (p.y, p.x)):
                    real_x = point.get_real_x()
                    real_y = point.get_real_y()
                    text_widget.insert(tk.END, f"  ({point.x:3d}, {point.y:3d}) -> ({real_x:6.2f}, {real_y:6.2f})\n")
                text_widget.insert(tk.END, "\n")
        except:
            pass
            
        # Get Boundary points
        try:
            points_B = list(self.current_area.points_list(types='B'))
            if points_B:
                text_widget.insert(tk.END, f"Boundary Points ({len(points_B)}): \n")
                for point in sorted(points_B, key=lambda p: (p.y, p.x)):
                    real_x = point.get_real_x()
                    real_y = point.get_real_y()
                    text_widget.insert(tk.END, f"  ({point.x:3d}, {point.y:3d}) -> ({real_x:6.2f}, {real_y:6.2f})\n")
                text_widget.insert(tk.END, "\n")
        except:
            pass
        
        text_widget.config(state=tk.DISABLED)


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = AreaVisualizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()