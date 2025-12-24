import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json
from datetime import datetime
import threading
import time
from collections import deque

class CameraCalibrationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV Camera Calibration Tool")
        self.root.geometry("1200x800")

        # Calibration data
        self.images = []
        self.image_points = []
        self.object_points = []
        self.captured_frames = []
        self.reprojection_errors = []
        self.calibration_result = None

        # Persistent dots for visualization
        self.persistent_dots = []  # Store all detections without limit

        # Camera settings
        self.camera = None
        self.camera_running = False
        self.timer_running = False
        self.timer_interval = 2.0

        # Calibration pattern (chessboard)
        self.pattern_size = (9, 6)  # internal corners
        self.square_size = 1.0  # in arbitrary units

        # Camera models
        self.camera_models = {
            "Pinhole (Standard)": "pinhole",
            "Pinhole with Rational": "rational",
            "Fisheye": "fisheye",
            "Pinhole Thin Prism": "thin_prism"
        }
        self.selected_model = "pinhole"

        # Create main container
        self.main_container = tk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.show_model_selection()

    def show_model_selection(self):
        self.clear_container()

        frame = tk.Frame(self.main_container)
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        tk.Label(frame, text="Camera Calibration Tool", font=("Arial", 24, "bold")).pack(pady=20)
        tk.Label(frame, text="Select Camera Model:", font=("Arial", 14)).pack(pady=10)

        self.model_var = tk.StringVar(value="Pinhole (Standard)")
        for model_name in self.camera_models.keys():
            tk.Radiobutton(frame, text=model_name, variable=self.model_var, 
                          value=model_name, font=("Arial", 12)).pack(anchor=tk.W, padx=50)

        tk.Label(frame, text="\nChessboard Pattern Settings:", font=("Arial", 14)).pack(pady=10)

        pattern_frame = tk.Frame(frame)
        pattern_frame.pack(pady=5)
        tk.Label(pattern_frame, text="Columns (internal corners):", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W)
        self.pattern_cols = tk.Spinbox(pattern_frame, from_=3, to=20, width=10, font=("Arial", 10))
        self.pattern_cols.delete(0, tk.END)
        self.pattern_cols.insert(0, "9")
        self.pattern_cols.grid(row=0, column=1, padx=5)

        tk.Label(pattern_frame, text="Rows (internal corners):", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W)
        self.pattern_rows = tk.Spinbox(pattern_frame, from_=3, to=20, width=10, font=("Arial", 10))
        self.pattern_rows.delete(0, tk.END)
        self.pattern_rows.insert(0, "6")
        self.pattern_rows.grid(row=1, column=1, padx=5)

        tk.Label(pattern_frame, text="Square size (mm):", font=("Arial", 10)).grid(row=2, column=0, sticky=tk.W)
        self.square_size_entry = tk.Entry(pattern_frame, width=10, font=("Arial", 10))
        self.square_size_entry.insert(0, "25.0")
        self.square_size_entry.grid(row=2, column=1, padx=5)

        button_frame = tk.Frame(frame)
        button_frame.pack(pady=30)

        tk.Button(button_frame, text="Generate Checkerboard", command=self.generate_checkerboard,
                 font=("Arial", 12), bg="#FF9800", fg="white", padx=15, pady=8).pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Start Calibration", command=self.start_calibration,
                 font=("Arial", 14), bg="#4CAF50", fg="white", padx=20, pady=10).pack(side=tk.LEFT, padx=5)

    def generate_checkerboard(self):
        """Generate and save a checkerboard pattern for printing"""
        try:
            cols = int(self.pattern_cols.get()) + 1
            rows = int(self.pattern_rows.get()) + 1
            square_size_mm = float(self.square_size_entry.get())

            dpi = 300
            mm_to_inch = 0.0393701
            square_size_px = int(square_size_mm * mm_to_inch * dpi)

            width_px = cols * square_size_px
            height_px = rows * square_size_px

            img = Image.new('RGB', (width_px, height_px), 'white')
            draw = ImageDraw.Draw(img)

            for i in range(rows):
                for j in range(cols):
                    if (i + j) % 2 == 0:
                        x1 = j * square_size_px
                        y1 = i * square_size_px
                        x2 = x1 + square_size_px
                        y2 = y1 + square_size_px
                        draw.rectangle([x1, y1, x2, y2], fill='black')

            border = int(0.5 * mm_to_inch * dpi)
            final_img = Image.new('RGB', (width_px + 2*border, height_px + 2*border), 'white')
            final_img.paste(img, (border, border))

            draw = ImageDraw.Draw(final_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()

            info_text = f"Checkerboard: {cols}x{rows} squares ({cols-1}x{rows-1} internal corners) | Square size: {square_size_mm}mm"
            draw.text((border, border//2), info_text, fill='black', font=font)

            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
                initialfile=f"checkerboard_{cols}x{rows}_{square_size_mm}mm.png"
            )

            if filename:
                if filename.endswith('.pdf'):
                    final_img.save(filename, "PDF", resolution=dpi)
                else:
                    final_img.save(filename, dpi=(dpi, dpi))

                messagebox.showinfo("Success", 
                    f"Checkerboard pattern saved!\n\n"
                    f"File: {filename}\n"
                    f"Pattern: {cols}x{rows} squares ({cols-1}x{rows-1} internal corners)\n"
                    f"Square size: {square_size_mm}mm\n"
                    f"Resolution: {dpi} DPI\n\n"
                    f"Print this at 100% scale (no scaling) for accurate calibration.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate checkerboard: {str(e)}")

    def start_calibration(self):
        self.selected_model = self.camera_models[self.model_var.get()]
        self.pattern_size = (int(self.pattern_cols.get()), int(self.pattern_rows.get()))
        self.square_size = float(self.square_size_entry.get())

        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return

        self.camera_running = True
        self.persistent_dots = []
        self.show_capture_view()

    def show_capture_view(self):
        self.clear_container()

        # Top controls
        control_frame = tk.Frame(self.main_container, bg="#333", height=60)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        control_frame.pack_propagate(False)

        tk.Button(control_frame, text="Capture Image", command=self.capture_single,
                 font=("Arial", 12), bg="#2196F3", fg="white", padx=15, pady=5).pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(control_frame, text="Timer interval (s):", bg="#333", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.timer_entry = tk.Entry(control_frame, width=5, font=("Arial", 10))
        self.timer_entry.insert(0, "2.0")
        self.timer_entry.pack(side=tk.LEFT, padx=5)

        self.timer_btn = tk.Button(control_frame, text="Start Timer", command=self.toggle_timer,
                                   font=("Arial", 12), bg="#FF9800", fg="white", padx=15, pady=5)
        self.timer_btn.pack(side=tk.LEFT, padx=10)

        tk.Label(control_frame, text="Images captured:", bg="#333", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        self.image_count_label = tk.Label(control_frame, text="0", bg="#333", fg="#4CAF50", font=("Arial", 14, "bold"))
        self.image_count_label.pack(side=tk.LEFT)

        tk.Button(control_frame, text="Clear Dots", command=self.clear_persistent_dots,
                 font=("Arial", 11), bg="#9C27B0", fg="white", padx=10, pady=5).pack(side=tk.LEFT, padx=10)

        tk.Button(control_frame, text="Review Images →", command=self.show_review_view,
                 font=("Arial", 12), bg="#4CAF50", fg="white", padx=15, pady=5).pack(side=tk.RIGHT, padx=10, pady=10)

        # Camera view
        self.camera_label = tk.Label(self.main_container, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.update_camera_feed()

    def clear_persistent_dots(self):
        """Clear all persistent dots from the display"""
        self.persistent_dots = []

    def update_camera_feed(self):
        if not self.camera_running:
            return

        ret, frame = self.camera.read()
        if ret:
            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret_corners, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

            current_corners = None
            dot_radius = 8

            if ret_corners:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Calculate dot radius based on corner spacing
                if len(corners_refined) > 1:
                    dist = np.linalg.norm(corners_refined[0] - corners_refined[1])
                    dot_radius = max(3, int(dist * 0.15))

                current_corners = corners_refined

                # Draw chessboard lines for current detection
                cv2.drawChessboardCorners(display_frame, self.pattern_size, corners_refined, ret_corners)

            # Draw all persistent dots (no fading)
            overlay = display_frame.copy()

            for dot_data in self.persistent_dots:
                x, y, radius = dot_data
                cv2.circle(overlay, (int(x), int(y)), radius, (0, 200, 0), -1)
                cv2.circle(overlay, (int(x), int(y)), 2, (255, 255, 255), -1)

            # Draw current detection corners on top with brighter color
            if current_corners is not None:
                for corner in current_corners:
                    x, y = corner.ravel()
                    cv2.circle(overlay, (int(x), int(y)), dot_radius, (0, 255, 0), -1)
                    cv2.circle(overlay, (int(x), int(y)), 2, (255, 255, 255), -1)

            # Blend overlay
            alpha_blend = 0.6
            display_frame = cv2.addWeighted(overlay, alpha_blend, display_frame, 1 - alpha_blend, 0)

            # Status text
            if ret_corners:
                text = f"Pattern detected! ({len(corners)} points)"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(display_frame, (5, 5), (text_width + 15, text_height + 15), (0, 255, 0), -1)
                cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                text = "No pattern detected - move checkerboard into view"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_frame, (5, 5), (text_width + 15, text_height + 15), (0, 0, 255), -1)
                cv2.putText(display_frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Convert to PhotoImage
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w = display_frame.shape[:2]
            max_width, max_height = 1180, 700
            scale = min(max_width/w, max_height/h)
            new_w, new_h = int(w*scale), int(h*scale)
            display_frame = cv2.resize(display_frame, (new_w, new_h))

            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

        self.root.after(10, self.update_camera_feed)

    def capture_single(self):
        if not self.camera_running:
            return

        ret, frame = self.camera.read()
        if ret:
            self.process_captured_frame(frame)

    def toggle_timer(self):
        if self.timer_running:
            self.timer_running = False
            self.timer_btn.config(text="Start Timer", bg="#FF9800")
        else:
            try:
                self.timer_interval = float(self.timer_entry.get())
                self.timer_running = True
                self.timer_btn.config(text="Stop Timer", bg="#F44336")
                threading.Thread(target=self.timer_capture, daemon=True).start()
            except ValueError:
                messagebox.showerror("Error", "Invalid timer interval!")

    def timer_capture(self):
        while self.timer_running and self.camera_running:
            time.sleep(self.timer_interval)
            if self.timer_running:
                self.root.after(0, self.capture_single)

    def process_captured_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            self.captured_frames.append(frame.copy())
            self.image_points.append(corners_refined)

            # Generate object points
            objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
            objp *= self.square_size
            self.object_points.append(objp)

            # Add corners to persistent dots
            if len(corners_refined) > 1:
                dist = np.linalg.norm(corners_refined[0] - corners_refined[1])
                dot_radius = max(3, int(dist * 0.15))
            else:
                dot_radius = 8

            for corner in corners_refined:
                x, y = corner.ravel()
                self.persistent_dots.append((x, y, dot_radius))

            self.image_count_label.config(text=str(len(self.captured_frames)))
        else:
            messagebox.showwarning("Warning", "No chessboard pattern detected in captured image!")

    def show_review_view(self):
        if len(self.captured_frames) < 3:
            messagebox.showwarning("Warning", "Please capture at least 3 images before reviewing!")
            return

        self.camera_running = False
        self.timer_running = False
        if self.camera:
            self.camera.release()

        self.clear_container()

        # Top bar
        top_bar = tk.Frame(self.main_container, bg="#333", height=50)
        top_bar.pack(side=tk.TOP, fill=tk.X)
        top_bar.pack_propagate(False)

        tk.Label(top_bar, text=f"Review Images ({len(self.captured_frames)} captured)", 
                bg="#333", fg="white", font=("Arial", 14, "bold")).pack(side=tk.LEFT, padx=20, pady=10)

        tk.Button(top_bar, text="← Back to Capture", command=self.return_to_capture,
                 font=("Arial", 11), bg="#757575", fg="white", padx=10, pady=5).pack(side=tk.RIGHT, padx=10)

        tk.Button(top_bar, text="Finish Calibration →", command=self.perform_calibration,
                 font=("Arial", 11), bg="#4CAF50", fg="white", padx=15, pady=5).pack(side=tk.RIGHT, padx=10)

        # Main content
        content_frame = tk.Frame(self.main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Image list
        left_frame = tk.Frame(content_frame, width=200, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="Captured Images", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=10)

        list_frame = tk.Frame(left_frame, bg="#f0f0f0")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("Arial", 10))
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)

        for i in range(len(self.captured_frames)):
            self.image_listbox.insert(tk.END, f"Image {i+1}")

        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        # Delete button below list
        tk.Button(left_frame, text="Delete Selected", command=self.delete_selected_image,
                 font=("Arial", 10), bg="#F44336", fg="white", padx=10, pady=5).pack(pady=10)

        # Right: Image display and error graph
        right_frame = tk.Frame(content_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.review_image_label = tk.Label(right_frame, bg="black")
        self.review_image_label.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.error_graph_label = tk.Label(right_frame, bg="white", height=8, relief=tk.SUNKEN)
        self.error_graph_label.pack(fill=tk.X)

        # Always show the graph - either with corner counts or reprojection errors
        self.update_error_graph()

        if len(self.captured_frames) > 0:
            self.image_listbox.selection_set(0)
            self.on_image_select(None)

    def delete_selected_image(self):
        """Delete the selected image from the calibration set"""
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an image to delete!")
            return

        idx = selection[0]

        # Confirm deletion
        if messagebox.askyesno("Confirm Delete", f"Delete Image {idx+1}?"):
            # Remove from all lists
            del self.captured_frames[idx]
            del self.image_points[idx]
            del self.object_points[idx]
            if self.reprojection_errors and idx < len(self.reprojection_errors):
                del self.reprojection_errors[idx]

            # Clear calibration result if we had one
            self.calibration_result = None

            # Refresh the review view
            if len(self.captured_frames) >= 3:
                self.show_review_view()
            else:
                messagebox.showinfo("Info", "Less than 3 images remaining. Returning to capture mode.")
                self.return_to_capture()

    def on_image_select(self, event):
        selection = self.image_listbox.curselection()
        if selection:
            idx = selection[0]
            frame = self.captured_frames[idx].copy()

            # Draw detected corners on the image
            if idx < len(self.image_points):
                corners = self.image_points[idx]

                # Calculate dot radius
                if len(corners) > 1:
                    dist = np.linalg.norm(corners[0] - corners[1])
                    dot_radius = max(3, int(dist * 0.15))
                else:
                    dot_radius = 8

                # Draw corners
                overlay = frame.copy()
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(overlay, (int(x), int(y)), dot_radius, (0, 255, 0), -1)
                    cv2.circle(overlay, (int(x), int(y)), 2, (255, 255, 255), -1)

                # Draw chessboard pattern
                cv2.drawChessboardCorners(frame, self.pattern_size, corners, True)

                # Blend
                alpha = 0.6
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Add info text
                info_text = f"Image {idx+1}: {len(corners)} corners detected"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # If we have reprojection error for this image, show it
                if self.reprojection_errors and idx < len(self.reprojection_errors):
                    error_text = f"Reprojection Error: {self.reprojection_errors[idx]:.4f} px"
                    cv2.putText(frame, error_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = display_frame.shape[:2]
            max_h = 500
            if h > max_h:
                scale = max_h / h
                display_frame = cv2.resize(display_frame, (int(w*scale), int(h*scale)))

            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.review_image_label.imgtk = imgtk
            self.review_image_label.configure(image=imgtk)

    def return_to_capture(self):
        self.camera = cv2.VideoCapture(0)
        self.camera_running = True
        self.show_capture_view()

    def perform_calibration(self):
        if len(self.captured_frames) < 3:
            messagebox.showerror("Error", "Need at least 3 images for calibration!")
            return

        # Get image size
        h, w = self.captured_frames[0].shape[:2]

        try:
            if self.selected_model == "fisheye":
                calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

                K = np.zeros((3, 3))
                D = np.zeros((4, 1))
                rvecs = []
                tvecs = []

                ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    self.object_points,
                    self.image_points,
                    (w, h),
                    K,
                    D,
                    rvecs,
                    tvecs,
                    calibration_flags,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                )

                self.reprojection_errors = []
                for i in range(len(self.object_points)):
                    imgpoints2, _ = cv2.fisheye.projectPoints(
                        self.object_points[i].reshape(-1, 1, 3),
                        rvecs[i],
                        tvecs[i],
                        K,
                        D
                    )
                    error = cv2.norm(self.image_points[i], imgpoints2.reshape(-1, 1, 2), cv2.NORM_L2) / len(imgpoints2)
                    self.reprojection_errors.append(error)

                self.calibration_result = {
                    "model": "fisheye",
                    "rms_error": ret,
                    "camera_matrix": K.tolist(),
                    "distortion_coefficients": D.tolist(),
                    "image_size": [w, h],
                    "reprojection_errors": self.reprojection_errors,
                    "avg_reprojection_error": np.mean(self.reprojection_errors)
                }

            else:
                flags = 0
                if self.selected_model == "rational":
                    flags = cv2.CALIB_RATIONAL_MODEL
                elif self.selected_model == "thin_prism":
                    flags = cv2.CALIB_THIN_PRISM_MODEL

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    self.object_points,
                    self.image_points,
                    (w, h),
                    None,
                    None,
                    flags=flags
                )

                self.reprojection_errors = []
                for i in range(len(self.object_points)):
                    imgpoints2, _ = cv2.projectPoints(
                        self.object_points[i],
                        rvecs[i],
                        tvecs[i],
                        mtx,
                        dist
                    )
                    error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    self.reprojection_errors.append(error)

                self.calibration_result = {
                    "model": self.selected_model,
                    "rms_error": ret,
                    "camera_matrix": mtx.tolist(),
                    "distortion_coefficients": dist.tolist(),
                    "image_size": [w, h],
                    "reprojection_errors": self.reprojection_errors,
                    "avg_reprojection_error": np.mean(self.reprojection_errors)
                }

            # Show the results view with full details
            self.show_results_view()

        except Exception as e:
            messagebox.showerror("Calibration Error", f"Error during calibration: {str(e)}")

    def update_error_graph(self):
        """Create bar graph - shows corner counts before calibration, errors after"""

        # Create bar graph
        fig_width = 700
        fig_height = 250
        img = np.ones((fig_height, fig_width, 3), dtype=np.uint8) * 255

        n_images = len(self.captured_frames)
        if n_images == 0:
            return

        margin_left = 60
        margin_right = 40
        margin_top = 50
        margin_bottom = 50

        graph_width = fig_width - margin_left - margin_right
        graph_height = fig_height - margin_top - margin_bottom

        bar_width = max(8, (graph_width // n_images) - 8)

        # Decide what to show
        if self.reprojection_errors:
            # Show reprojection errors
            data = self.reprojection_errors
            max_val = max(data)
            avg_val = np.mean(data)
            title = "Reprojection Error per Image (pixels)"
            ylabel = "Error (px)"
            show_avg_line = True
        else:
            # Show number of corners detected
            data = [len(pts) for pts in self.image_points]
            max_val = max(data) if data else 1
            avg_val = np.mean(data) if data else 0
            title = "Detected Corners per Image"
            ylabel = "Corners"
            show_avg_line = False

        # Draw axes
        cv2.line(img, (margin_left, margin_top), (margin_left, fig_height - margin_bottom), (0, 0, 0), 2)
        cv2.line(img, (margin_left, fig_height - margin_bottom), (fig_width - margin_right, fig_height - margin_bottom), (0, 0, 0), 2)

        # Draw bars
        for i, val in enumerate(data):
            bar_height = int((val / (max_val * 1.1)) * graph_height)
            x_center = margin_left + (i + 0.5) * (graph_width / n_images)
            x1 = int(x_center - bar_width // 2)
            x2 = int(x_center + bar_width // 2)
            y1 = fig_height - margin_bottom - bar_height
            y2 = fig_height - margin_bottom

            if self.reprojection_errors:
                # Color bars based on error (green to red gradient)
                error_ratio = val / max_val
                color = (int(50 * (1 - error_ratio)), int(150 + 105 * (1 - error_ratio)), int(255 * (1 - error_ratio)))
            else:
                # Blue bars for corner counts
                color = (66, 133, 244)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)

            # Image number below bar
            text = str(i+1)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            text_x = int(x_center - text_size[0] // 2)
            cv2.putText(img, text, (text_x, fig_height - margin_bottom + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Draw average line (only for reprojection errors)
        if show_avg_line:
            avg_y = fig_height - margin_bottom - int((avg_val / (max_val * 1.1)) * graph_height)
            cv2.line(img, (margin_left, avg_y), (fig_width - margin_right, avg_y), (244, 67, 54), 2)
            cv2.putText(img, "Avg", (margin_left - 35, avg_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (244, 67, 54), 1)

        # Y-axis labels
        for i in range(5):
            y_val = (max_val * 1.1) * i / 4
            y_pos = fig_height - margin_bottom - int((y_val / (max_val * 1.1)) * graph_height)
            cv2.line(img, (margin_left - 5, y_pos), (margin_left, y_pos), (0, 0, 0), 1)
            if self.reprojection_errors:
                cv2.putText(img, f"{y_val:.2f}", (5, y_pos + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            else:
                cv2.putText(img, f"{int(y_val)}", (10, y_pos + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Title
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        title_x = (fig_width - title_size[0]) // 2
        cv2.putText(img, title, (title_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Stats
        if self.reprojection_errors:
            stats_text = f"Avg: {avg_val:.4f} | Max: {max(data):.4f} | Min: {min(data):.4f}"
        else:
            stats_text = f"Avg: {avg_val:.1f} | Max: {max(data)} | Min: {min(data)} corners"
        cv2.putText(img, stats_text, (margin_left + 10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)

        self.error_graph_label.imgtk = imgtk
        self.error_graph_label.configure(image=imgtk, text="")

    def show_results_view(self):
        self.clear_container()

        # Top bar
        top_bar = tk.Frame(self.main_container, bg="#4CAF50", height=60)
        top_bar.pack(side=tk.TOP, fill=tk.X)
        top_bar.pack_propagate(False)

        tk.Label(top_bar, text="✓ Calibration Complete!", 
                bg="#4CAF50", fg="white", font=("Arial", 18, "bold")).pack(side=tk.LEFT, padx=20, pady=15)

        button_container = tk.Frame(top_bar, bg="#4CAF50")
        button_container.pack(side=tk.RIGHT, padx=20)

        tk.Button(button_container, text="← Back to Review", command=self.show_review_view,
                 font=("Arial", 11), bg="white", fg="#757575", padx=15, pady=8).pack(side=tk.LEFT, padx=5)

        tk.Button(button_container, text="Save Sample Code", command=self.save_sample_code,
                 font=("Arial", 11), bg="white", fg="#2196F3", padx=15, pady=8).pack(side=tk.LEFT, padx=5)

        tk.Button(button_container, text="Save as JSON", command=self.save_calibration,
                 font=("Arial", 11), bg="white", fg="#4CAF50", padx=15, pady=8).pack(side=tk.LEFT, padx=5)

        # Results content
        results_frame = tk.Frame(self.main_container)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # Scrollable text widget
        text_frame = tk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_text = tk.Text(text_frame, wrap=tk.WORD, font=("Courier", 10), 
                                   yscrollcommand=scrollbar.set, bg="#f9f9f9")
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)

        # Populate results
        result = self.calibration_result
        results_str = f"""
CALIBRATION RESULTS
{'=' * 80}

Camera Model: {result['model'].upper()}
RMS Reprojection Error: {result['rms_error']:.6f} pixels
Average Reprojection Error: {result['avg_reprojection_error']:.6f} pixels
Number of Images: {len(self.captured_frames)}
Image Size: {result['image_size'][0]} x {result['image_size'][1]}

{'=' * 80}
CAMERA MATRIX (K)
{'=' * 80}
"""

        K = np.array(result['camera_matrix'])
        results_str += f"[{K[0, 0]:12.6f}  {K[0, 1]:12.6f}  {K[0, 2]:12.6f}]\n"
        results_str += f"[{K[1, 0]:12.6f}  {K[1, 1]:12.6f}  {K[1, 2]:12.6f}]\n"
        results_str += f"[{K[2, 0]:12.6f}  {K[2, 1]:12.6f}  {K[2, 2]:12.6f}]\n\n"

        results_str += f"Focal Length (fx, fy): ({K[0, 0]:.2f}, {K[1, 1]:.2f})\n"
        results_str += f"Principal Point (cx, cy): ({K[0, 2]:.2f}, {K[1, 2]:.2f})\n\n"

        results_str += f"{'=' * 80}\n"
        results_str += f"DISTORTION COEFFICIENTS\n"
        results_str += f"{'=' * 80}\n"

        dist = np.array(result['distortion_coefficients']).flatten()
        if result['model'] == 'fisheye':
            results_str += f"k1: {dist[0]:.8f}\n"
            results_str += f"k2: {dist[1]:.8f}\n"
            results_str += f"k3: {dist[2]:.8f}\n"
            results_str += f"k4: {dist[3]:.8f}\n"
        else:
            coeff_names = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2', 's3', 's4']
            for i, val in enumerate(dist):
                if i < len(coeff_names):
                    results_str += f"{coeff_names[i]}: {val:.8f}\n"

        results_str += f"\n{'=' * 80}\n"
        results_str += f"PER-IMAGE REPROJECTION ERRORS\n"
        results_str += f"{'=' * 80}\n"

        for i, error in enumerate(result['reprojection_errors']):
            results_str += f"Image {i+1:2d}: {error:.4f} pixels\n"

        self.results_text.insert('1.0', results_str)
        self.results_text.config(state=tk.DISABLED)

        # Bottom button frame
        button_frame = tk.Frame(results_frame)
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="Start New Calibration", command=self.reset_calibration,
                 font=("Arial", 12), bg="#2196F3", fg="white", padx=20, pady=10).pack(side=tk.LEFT, padx=10)

    def save_calibration(self):
        if not self.calibration_result:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"camera_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        if filename:
            with open(filename, 'w') as f:
                json.dump(self.calibration_result, f, indent=4)
            messagebox.showinfo("Success", f"Calibration saved to:\n{filename}")

    
    def save_sample_code(self):
        """Generate and save sample Python code using the calibration data"""
        if not self.calibration_result:
            return

        result = self.calibration_result

        # Generate sample code based on camera model
        if result['model'] == 'fisheye':
            sample_code = f"""#!/usr/bin/env python3
\"\"\"
Sample code for using fisheye camera calibration
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Camera Model: {result['model']}
RMS Error: {result['rms_error']:.6f} pixels
\"\"\"

import cv2
import numpy as np

# Camera calibration parameters
camera_matrix = np.array({result['camera_matrix']})
dist_coeffs = np.array({result['distortion_coefficients']})
image_size = {tuple(result['image_size'])}

def undistort_image(img):
    \"\"\"Undistort a fisheye image\"\"\"
    h, w = img.shape[:2]

    # Calculate new camera matrix for fisheye
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix, dist_coeffs, (w, h), np.eye(3), balance=0.0
    )

    # Generate undistortion maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2
    )

    # Apply undistortion
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted

def undistort_points(points):
    \"\"\"Undistort image points (Nx2 array)\"\"\"
    points = points.reshape(-1, 1, 2).astype(np.float32)
    undistorted = cv2.fisheye.undistortPoints(
        points, camera_matrix, dist_coeffs, P=camera_matrix
    )
    return undistorted.reshape(-1, 2)

# Example usage
if __name__ == "__main__":
    # Load an image
    img = cv2.imread('your_image.jpg')

    if img is not None:
        # Undistort the image
        undistorted_img = undistort_image(img)

        # Display results
        cv2.imshow('Original', img)
        cv2.imshow('Undistorted', undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save undistorted image
        cv2.imwrite('undistorted_output.jpg', undistorted_img)

    # Example: Undistort specific points
    # distorted_points = np.array([[320, 240], [640, 480]], dtype=np.float32)
    # undistorted_points = undistort_points(distorted_points)
    # print("Undistorted points:", undistorted_points)
"""
        else:
            sample_code = f"""#!/usr/bin/env python3
\"\"\"
Sample code for using camera calibration
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Camera Model: {result['model']}
RMS Error: {result['rms_error']:.6f} pixels
\"\"\"

import cv2
import numpy as np

# Camera calibration parameters
camera_matrix = np.array({result['camera_matrix']})
dist_coeffs = np.array({result['distortion_coefficients']})
image_size = {tuple(result['image_size'])}

def undistort_image(img, alpha=1.0):
    \"\"\"
    Undistort an image

    Args:
        img: Input distorted image
        alpha: Free scaling parameter (0-1)
               0 = all pixels valid but cropped
               1 = all source pixels retained but with black borders
    \"\"\"
    h, w = img.shape[:2]

    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
    )

    # Undistort
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop to region of interest if alpha=0
    if alpha == 0:
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

    return undistorted

def undistort_image_remap(img, alpha=1.0):
    \"\"\"Undistort using remap (more efficient for multiple frames)\"\"\"
    h, w = img.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
    )

    # Generate undistortion maps (do this once, reuse for all frames)
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
    )

    # Apply remapping
    undistorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    if alpha == 0:
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

    return undistorted

def undistort_points(points):
    \"\"\"Undistort image points (Nx2 array)\"\"\"
    points = points.reshape(-1, 1, 2).astype(np.float32)
    undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted.reshape(-1, 2)

def project_3d_to_2d(object_points, rvec, tvec):
    \"\"\"Project 3D points to 2D image coordinates\"\"\"
    image_points, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    return image_points.reshape(-1, 2)

# Example usage
if __name__ == "__main__":
    # Load an image
    img = cv2.imread('your_image.jpg')

    if img is not None:
        # Undistort the image (alpha=1 keeps all pixels, alpha=0 crops to valid region)
        undistorted_img = undistort_image(img, alpha=1.0)

        # Display results
        cv2.imshow('Original', img)
        cv2.imshow('Undistorted', undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save undistorted image
        cv2.imwrite('undistorted_output.jpg', undistorted_img)

    # Example: Undistort specific points
    # distorted_points = np.array([[320, 240], [640, 480]], dtype=np.float32)
    # undistorted_points = undistort_points(distorted_points)
    # print("Undistorted points:", undistorted_points)

    # Example: Video undistortion
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     undistorted_frame = undistort_image(frame)
    #     cv2.imshow('Undistorted Video', undistorted_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            initialfile=f"calibration_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        )

        if filename:
            with open(filename, 'w') as f:
                f.write(sample_code)
            messagebox.showinfo("Success", 
                f"Sample code saved to:\n{filename}\n\n"
                f"This code includes functions for:\n"
                f"- Image undistortion\n"
                f"- Point undistortion\n"
                f"- 3D to 2D projection\n"
                f"- Video processing examples")

    def reset_calibration(self):
        self.images = []
        self.image_points = []
        self.object_points = []
        self.captured_frames = []
        self.reprojection_errors = []
        self.calibration_result = None
        self.persistent_dots = []

        self.show_model_selection()

    def clear_container(self):
        for widget in self.main_container.winfo_children():
            widget.destroy()

    def on_closing(self):
        self.camera_running = False
        self.timer_running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraCalibrationTool(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
