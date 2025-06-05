import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QComboBox, QLineEdit, QCheckBox, QTextEdit,
                               QScrollArea, QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene,
                               QGraphicsPixmapItem)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
import json
import os
import csv
import time
import threading
import gc
import sys
from concurrent.futures import ThreadPoolExecutor

# === Zoomable Graphics View ===
ZOOM_STEP = 0.1
MIN_ZOOM = 0.5
MAX_ZOOM = 3.0

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.zoom_level = 1.0
        self.pixmap_item = None
        self.initialized = False
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_pixmap(self, pixmap):
        if self.pixmap_item:
            self.scene().removeItem(self.pixmap_item)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        if not self.initialized:
            self.fit_to_view()
            self.initialized = True
        else:
            self.resetTransform()
            self.scale(self.zoom_level, self.zoom_level)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            zoom_factor = 1.0 + ZOOM_STEP if delta > 0 else 1.0 - ZOOM_STEP
            new_zoom = self.zoom_level * zoom_factor
            if MIN_ZOOM <= new_zoom <= MAX_ZOOM:
                self.zoom_level = new_zoom
                self.resetTransform()
                self.scale(self.zoom_level, self.zoom_level)
        else:
            super().wheelEvent(event)

    def fit_to_view(self):
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

# === Configuration Loading ===
def load_config(config_file):
    default_config = {
        "canny_thresh1": 150.0,
        "canny_thresh2": 350.0,
        "min_area": 0,
        "max_area": 1000000,
        "exposure": -14.0
    }
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            default_config.update(config)
        return default_config
    except FileNotFoundError:
        return None
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to load config JSON: {str(e)}")
        return None

# === Load ROI Coordinates and Warp Points from JSON ===
def load_rois(rois_file):
    try:
        with open(rois_file, 'r') as f:
            data = json.load(f)
        
        # Validate warp points
        if 'warp_points' not in data or not isinstance(data['warp_points'], list):
            raise ValueError("JSON must contain a 'warp_points' list")
        warp_points = []
        if data['warp_points']:
            if len(data['warp_points']) != 4:
                raise ValueError("warp_points must be empty or contain exactly 4 points")
            for point in data['warp_points']:
                if not all(key in point for key in ['x', 'y']):
                    raise ValueError(f"Invalid warp point: {point}")
                if not all(isinstance(point[key], (int, float)) for key in ['x', 'y']):
                    raise ValueError(f"Warp point coordinates must be numbers: {point}")
                warp_points.append((float(point['x']), float(point['y'])))
        
        # Validate ROIs
        if 'rois' not in data or not isinstance(data['rois'], list):
            raise ValueError("JSON must contain a 'rois' list")
        rois = []
        for roi in data['rois']:
            if not all(key in roi for key in ['left', 'right', 'top', 'bottom']):
                print(f"Skipping ROI due to missing keys: {roi}")
                continue
            x = roi['left']
            y = roi['top']
            w = roi['right'] - roi['left']
            h = roi['bottom'] - roi['top']
            print(f"Calculated ROI: x={x}, y={y}, w={w}, h={h}")
            if w <= 0 or h <= 0:
                print(f"Skipping invalid ROI (w={w}, h={h}): {roi}")
                continue
            rows = roi.get('rows', 1)
            v_gap = roi.get('v_gap', 0)
            if not isinstance(rows, int) or rows <= 0:
                rows = 1
            if not isinstance(v_gap, (int, float)) or v_gap < 0:
                v_gap = 0
            rois.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'rows': rows,
                'v_gap': v_gap
            })
        if not rois:
            QMessageBox.warning(None, "Warning", f"No valid ROIs found in {rois_file}")
            return [], {}, []
        rois.sort(key=lambda r: r['x'])
        global_row_map = {}
        current_row = 1
        for idx, roi in enumerate(rois):
            roi['roi_number'] = idx + 1
            roi_rows = roi['rows']
            for row_idx in range(roi_rows):
                global_row_map[current_row] = (idx, row_idx)
                current_row += 1
        return rois, global_row_map, warp_points
    except FileNotFoundError:
        QMessageBox.critical(None, "Error", f"ROI file {rois_file} not found!")
        return [], {}, []
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to load ROI JSON: {str(e)}")
        return [], {}, []

# === Folder Safety Check ===
def check_folder_safety(folder_name):
    base_path = os.path.join("experiments", folder_name)
    csv_path = os.path.join(base_path, "logs", "drosophila_spots.csv")
    log_path = os.path.join(base_path, "logs", "experiment_log.txt")
    
    if os.path.exists(base_path):
        return False, f"Folder '{base_path}' already exists! Choose a different folder name."
    if os.path.exists(csv_path):
        return False, f"CSV file '{csv_path}' already exists! Choose a different folder name."
    if os.path.exists(log_path):
        return False, f"Log file '{log_path}' already exists! Choose a different folder name."
    return True, ""

# === Log File Writing ===
def write_log_entry(log_file, message):
    try:
        with open(log_file, 'a') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
        return message
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to write to log: {str(e)}")
        return f"Error: Failed to write to log: {str(e)}"

# === Camera Handling ===
class CameraHandler:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = None
        self.connected = False

    def initialize(self, max_retries=3, retry_delay=5):
        for attempt in range(max_retries):
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            if self.cap.isOpened():
                self.connected = True
                return True
            time.sleep(retry_delay)
        QMessageBox.critical(None, "Error", "Camera not found!")
        self.connected = False
        return False

    def capture(self, max_retries=3, retry_delay=5):
        if not self.connected or not self.cap:
            return None
        for attempt in range(max_retries):
            ret, frame = self.cap.read()
            if ret:
                return frame
            time.sleep(retry_delay)
            self.cap.release()
            if not self.initialize(max_retries=1):
                break
        QMessageBox.critical(None, "Error", "Camera disconnected!")
        self.connected = False
        return None

    def release(self):
        if self.cap:
            self.cap.release()
        self.connected = False
        self.cap = None
        gc.collect()

# === Detection and Processing ===
def process_roi(roi, x, y, config, rows, v_gap, roi_h):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    row_coordinates = [[] for _ in range(rows)]
    row_height = (roi_h - (rows - 1) * v_gap) / rows
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if config['min_area'] < area < config['max_area']:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.5:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        for row in range(rows):
                            row_top = row * (row_height + v_gap)
                            row_bottom = min(row_top + row_height, roi_h)
                            if row_top <= cY < row_bottom:
                                row_coordinates[row].append((cX, cY))
                                break
    return row_coordinates, x, y

def detect_spots_and_save(frame, timestamp, rois, global_row_map, config, save_real, save_processed, paths, warp_points, scale_factor=0.5):
    try:
        h, w = frame.shape[:2]
        
        # Apply perspective warp if warp_points are provided
        working_frame = frame.copy()
        if warp_points:
            src_points = np.float32(warp_points)
            # Define destination points as a rectangle (same size as input image)
            dst_points = np.float32([
                [0, 0],
                [w-1, 0],
                [w-1, h-1],
                [0, h-1]
            ])
            # Compute perspective transform
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            # Warp the image
            working_frame = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)
        
        small_frame = cv2.resize(working_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        preview_image = working_frame.copy()
        roi_coordinates = []
        scaled_rois = []
        for roi in rois:
            x, y, rw, rh = roi['x'], roi['y'], roi['w'], roi['h']
            if not all(isinstance(v, (int, float)) for v in [x, y, rw, rh]):
                error_msg = f"Invalid ROI coordinate types: {roi}"
                write_log_entry(paths['log_file'], error_msg)
                continue
            x, y, rw, rh = int(x), int(y), int(rw), int(rh)
            if rw <= 0 or rh <= 0:
                error_msg = f"Invalid ROI dimensions: w={rw}, h={rh} in {roi}"
                write_log_entry(paths['log_file'], error_msg)
                continue
            if x + rw > w or y + rh > h:
                error_msg = f"ROI {roi} exceeds image dimensions: {w}x{h}"
                write_log_entry(paths['log_file'], error_msg)
                continue
            scaled_rois.append((int(x * scale_factor), int(y * scale_factor),
                              int(rw * scale_factor), int(rh * scale_factor)))
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_roi, small_frame[y:y+h, x:x+w], x, y, config,
                                      roi['rows'], roi['v_gap'], h)
                       for roi, (x, y, w, h) in zip(rois, scaled_rois)]
            for idx, future in enumerate(futures):
                row_coords, x, y = future.result()
                scaled_row_coords = []
                for row in row_coords:
                    scaled_coords = [(int(cX/scale_factor), int(cY/scale_factor)) for cX, cY in row]
                    scaled_row_coords.append(scaled_coords)
                    for cX, cY in scaled_coords:
                        cv2.circle(preview_image, (int(x/scale_factor) + cX, int(y/scale_factor) + cY),
                                  5, (0, 0, 255), -1)
                roi_coordinates.append(scaled_row_coords)
        
        global_coords = [[] for _ in range(max(global_row_map.keys(), default=0))]
        for global_row, (roi_idx, row_idx) in global_row_map.items():
            global_coords[global_row - 1] = roi_coordinates[roi_idx][row_idx]
        
        current_row = 1
        for roi in rois:
            x, y, rw, rh = roi['x'], roi['y'], roi['w'], roi['h']
            x, y, rw, rh = int(x), int(y), int(rw), int(rh)
            rows, v_gap, roi_number = roi['rows'], roi['v_gap'], roi['roi_number']
            if rw <= 0 or rh <= 0:
                error_msg = f"Skipping invalid ROI {roi_number}: w={rw}, h={rh}"
                write_log_entry(paths['log_file'], error_msg)
                continue
            # Draw ROI rectangle with thicker border for clarity
            cv2.rectangle(preview_image, (x, y), (x + rw, y + rh), (0, 255, 0), 3)
            if rows > 1:
                row_height = (rh - (rows - 1) * v_gap) / rows
                for i in range(rows):
                    y_line = int(y + i * (row_height + v_gap))
                    cv2.line(preview_image, (x, y_line), (x + rw, y_line), (0, 255, 0), 1)
                    # Label each row with "flyX" where X is the global row number
                    cv2.putText(preview_image, f"fly{current_row}", (x + rw - 50, y_line + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    current_row += 1
            else:
                cv2.putText(preview_image, f"fly{current_row}", (x + rw - 50, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                current_row += 1
        
        if save_real:
            real_path = os.path.join(paths['captured'], f"image_{timestamp}.png")
            cv2.imwrite(real_path, frame)  # Save original (unwarped) frame
        if save_processed:
            processed_path = os.path.join(paths['processed'], f"processed_{timestamp}.png")
            cv2.imwrite(processed_path, preview_image)
        
        return global_coords, preview_image
    except Exception as e:
        error_msg = f"Failed to process image: {str(e)}"
        if 'log_file' in paths:
            write_log_entry(paths['log_file'], error_msg)
        return [[] for _ in range(max(global_row_map.keys(), default=0))], frame

def save_coordinates_to_csv_batch(buffer, output_csv, log_file, total_rows):
    try:
        # Check if CSV file exists; if not, write headers
        file_exists = os.path.exists(output_csv)
        with open(output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                headers = ['frames'] + [f'fly{i+1}' for i in range(total_rows)]
                writer.writerow(headers)
            for timestamp, global_coords in buffer:
                row = []
                for coords in global_coords:
                    if coords:
                        spot_string = ', '.join([f"({x}, {y})" for x, y in coords])
                        row.append(spot_string)
                    else:
                        row.append('')
                row.insert(0, timestamp)
                writer.writerow(row)
        return row
    except Exception as e:
        error_msg = f"Failed to write to CSV: {str(e)}"
        write_log_entry(log_file, error_msg)
        return []

# === GUI ===
class FlyDetectionGUI(QMainWindow):
    update_csv_signal = Signal(str)
    update_log_signal = Signal(str)
    update_image_signal = Signal(QPixmap)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fly Detection System")
        self.default_config = {
            "canny_thresh1": 150.0,
            "canny_thresh2": 350.0,
            "min_area": 0,
            "max_area": 1000000,
            "exposure": -14.0
        }
        self.config = self.default_config.copy()
        self.rois = []
        self.global_row_map = {}
        self.warp_points = []
        self.total_rows = 0
        self.camera = None
        self.running = False
        self.paths = {}
        self.coordinate_buffer = []
        self.rois_file = None
        self.config_file = None
        self.setup_gui()
        self.showMaximized()

    def setup_gui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(10, 10, 10, 10)

        scroll_area = QScrollArea()
        scroll_area.setWidget(left_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(420)
        main_layout.addWidget(scroll_area)

        left_layout.addWidget(QLabel("Experiment Status", font=("Arial", 12)))
        self.experiment_status_label = QLabel("Stopped")
        self.experiment_status_label.setStyleSheet("color: red")
        left_layout.addWidget(self.experiment_status_label)

        left_layout.addWidget(QLabel("Camera Status", font=("Arial", 12)))
        self.camera_status_label = QLabel("Disconnected")
        self.camera_status_label.setStyleSheet("color: red")
        left_layout.addWidget(self.camera_status_label)

        left_layout.addWidget(QLabel("Camera Index:"))
        self.camera_index_combo = QComboBox()
        self.camera_index_combo.addItems(["0", "1", "2", "3", "4"])
        self.camera_index_combo.setCurrentText("1")
        left_layout.addWidget(self.camera_index_combo)

        left_layout.addWidget(QLabel("Experiment Folder Name:"))
        self.folder_edit = QLineEdit("experiment_1")
        left_layout.addWidget(self.folder_edit)

        left_layout.addWidget(QLabel("Capture Interval (seconds):"))
        self.interval_edit = QLineEdit("3")
        left_layout.addWidget(self.interval_edit)

        left_layout.addWidget(QLabel("Experiment Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Manual", "Automatic"])
        self.mode_combo.currentTextChanged.connect(self.update_mode_ui)
        left_layout.addWidget(self.mode_combo)

        self.duration_widget = QWidget()
        duration_layout = QHBoxLayout(self.duration_widget)
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.addWidget(QLabel("Duration:"))
        self.hours_edit = QLineEdit("0")
        self.hours_edit.setFixedWidth(50)
        duration_layout.addWidget(self.hours_edit)
        duration_layout.addWidget(QLabel("h"))
        self.minutes_edit = QLineEdit("0")
        self.minutes_edit.setFixedWidth(50)
        duration_layout.addWidget(self.minutes_edit)
        duration_layout.addWidget(QLabel("m"))
        duration_layout.addStretch()
        left_layout.addWidget(self.duration_widget)
        self.duration_widget.setVisible(False)

        self.save_real_cb = QCheckBox("Save Real Images")
        self.save_real_cb.setChecked(True)
        left_layout.addWidget(self.save_real_cb)
        self.save_processed_cb = QCheckBox("Save Processed Images")
        self.save_processed_cb.setChecked(True)
        left_layout.addWidget(self.save_processed_cb)

        upload_roi_btn = QPushButton("Upload ROI JSON")
        upload_roi_btn.clicked.connect(self.upload_rois_json)
        left_layout.addWidget(upload_roi_btn)
        upload_config_btn = QPushButton("Upload Detection Params JSON")
        upload_config_btn.clicked.connect(self.upload_detection_params_json)
        left_layout.addWidget(upload_config_btn)

        self.start_btn = QPushButton("Start Experiment")
        self.start_btn.clicked.connect(self.start_experiment)
        left_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop Experiment")
        self.stop_btn.clicked.connect(self.stop_experiment)
        self.stop_btn.setEnabled(False)
        left_layout.addWidget(self.stop_btn)
        exit_btn = QPushButton("Save and Exit")
        exit_btn.clicked.connect(self.save_and_exit)
        left_layout.addWidget(exit_btn)

        left_layout.addWidget(QLabel("Latest CSV Entry:"))
        self.csv_text = QTextEdit()
        self.csv_text.setFixedHeight(50)
        self.csv_text.setReadOnly(True)
        left_layout.addWidget(self.csv_text)

        left_layout.addWidget(QLabel("Latest Log Entry:"))
        self.log_text = QTextEdit()
        self.log_text.setFixedHeight(50)
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text)

        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.right_panel)

        self.image_view = ZoomableGraphicsView(self.right_panel)
        right_layout.addWidget(self.image_view)

        self.update_csv_signal.connect(self.update_csv_text)
        self.update_log_signal.connect(self.update_log_text)
        self.update_image_signal.connect(self.update_image)

        self.update_status_indicators()

    @Slot(str)
    def update_csv_text(self, text):
        self.csv_text.setPlainText(text)

    @Slot(str)
    def update_log_text(self, text):
        self.log_text.setPlainText(text)

    @Slot(QPixmap)
    def update_image(self, pixmap):
        self.image_view.set_pixmap(pixmap)

    def update_mode_ui(self, mode):
        self.duration_widget.setVisible(mode == "Automatic")

    def upload_rois_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open ROI JSON File", "",
                                                   "JSON Files (*.json)")
        if file_path:
            try:
                self.rois, self.global_row_map, self.warp_points = load_rois(file_path)
                self.total_rows = max(self.global_row_map.keys(), default=0)
                self.rois_file = file_path
                if self.rois:
                    warp_status = "no warp points" if not self.warp_points else f"{len(self.warp_points)} warp points"
                    QMessageBox.information(self, "Success", f"Loaded {len(self.rois)} ROIs with {self.total_rows} total rows and {warp_status} from {file_path}")
                else:
                    self.rois = []
                    self.global_row_map = {}
                    self.warp_points = []
                    self.total_rows = 0
                    self.rois_file = None
                    QMessageBox.warning(self, "Warning", "No valid ROIs loaded.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load ROI JSON: {str(e)}")

    def upload_detection_params_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Detection Params JSON File", "",
                                                   "JSON Files (*.json)")
        if file_path:
            try:
                config = load_config(file_path)
                if config:
                    self.config = config
                    self.config_file = file_path
                    QMessageBox.information(self, "Success", f"Loaded detection parameters from {file_path}")
                else:
                    self.config = self.default_config.copy()
                    self.config_file = None
                    QMessageBox.information(self, "Info", f"Failed to load {file_path}. Using default parameters.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load config JSON: {str(e)}")

    def update_status_indicators(self):
        camera_status = "Connected" if self.camera and self.camera.connected else "Disconnected"
        camera_color = "green" if self.camera and self.camera.connected else "red"
        self.camera_status_label.setText(camera_status)
        self.camera_status_label.setStyleSheet(f"color: {camera_color}")

        experiment_status = "Running" if self.running else "Stopped"
        experiment_color = "green" if self.running else "red"
        self.experiment_status_label.setText(experiment_status)
        self.experiment_status_label.setStyleSheet(f"color: {experiment_color}")

        QTimer.singleShot(1000, self.update_status_indicators)

    def start_experiment(self):
        if self.running:
            QMessageBox.information(self, "Info", "Experiment already running!")
            return

        if not self.rois_file or not self.rois:
            QMessageBox.critical(self, "Error", "Please upload ROI JSON file!")
            return

        if self.warp_points and len(self.warp_points) != 4:
            QMessageBox.critical(self, "Error", "Warp points must be empty or contain exactly 4 points!")
            return

        if not self.config_file:
            self.config = self.default_config.copy()

        folder_name = self.folder_edit.text().strip()
        if not folder_name:
            QMessageBox.critical(self, "Error", "Please enter a valid folder name!")
            return
        is_safe, error_message = check_folder_safety(folder_name)
        if not is_safe:
            QMessageBox.critical(self, "Error", error_message)
            return

        try:
            interval = float(self.interval_edit.text())
            if interval <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid capture interval!")
            return

        duration = None
        if self.mode_combo.currentText() == "Automatic":
            try:
                hours = int(self.hours_edit.text())
                minutes = int(self.minutes_edit.text())
                if hours < 0 or minutes < 0 or (hours == 0 and minutes == 0):
                    raise ValueError
                duration = hours * 3600 + minutes * 60
            except ValueError:
                QMessageBox.critical(self, "Error", "Invalid duration for Automatic mode!")
                return

        base_path = os.path.join("experiments", folder_name)
        self.paths = {
            'captured': os.path.join(base_path, "captured"),
            'processed': os.path.join(base_path, "processed"),
            'logs': os.path.join(base_path, "logs"),
            'csv_file': os.path.join(base_path, "logs", "drosophila_spots.csv"),
            'log_file': os.path.join(base_path, "logs", "experiment_log.txt")
        }
        for path in [self.paths['captured'], self.paths['processed'], self.paths['logs']]:
            os.makedirs(path, exist_ok=True)

        try:
            with open(self.paths['log_file'], 'w') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Log initialized\n")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize log: {str(e)}")
            return

        try:
            camera_index = int(self.camera_index_combo.currentText())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid camera index!")
            return
        self.camera = CameraHandler(camera_index)
        if not self.camera.initialize():
            self.camera = None
            return

        self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment started"))

        self.camera_index_combo.setEnabled(False)
        self.folder_edit.setEnabled(False)
        self.interval_edit.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.hours_edit.setEnabled(False)
        self.minutes_edit.setEnabled(False)
        self.save_real_cb.setEnabled(False)
        self.save_processed_cb.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.running = True
        threading.Thread(target=self.run_experiment, args=(interval, duration), daemon=True).start()

    def run_experiment(self, interval, duration):
        start_time = time.time()
        timestamp_counter = 0
        while self.running:
            try:
                if duration and (time.time() - start_time) >= duration:
                    self.running = False
                    self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment stopped (duration reached)"))
                    break
                timestamp_counter += 1
                timestamp = f"frame_{timestamp_counter}"
                frame = self.camera.capture()
                if frame is None:
                    self.running = False
                    self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment stopped (camera failure)"))
                    break

                self.update_log_signal.emit(write_log_entry(self.paths['log_file'], f"Frame {timestamp} captured"))

                coords, processed_image = detect_spots_and_save(
                    frame, timestamp, self.rois, self.global_row_map, self.config,
                    self.save_real_cb.isChecked(), self.save_processed_cb.isChecked(), self.paths,
                    self.warp_points
                )
                self.coordinate_buffer.append((timestamp, coords))

                if self.coordinate_buffer:
                    csv_row = save_coordinates_to_csv_batch(self.coordinate_buffer, self.paths['csv_file'], 
                                                          self.paths['log_file'], self.total_rows)
                    if csv_row:
                        self.update_csv_signal.emit(str(csv_row))
                    self.coordinate_buffer = []

                try:
                    max_width = self.width() - 420
                    max_height = self.height() - 20
                    h, w = processed_image.shape[:2]
                    scale = min(max_width/w, max_height/h)
                    new_w, new_h = int(w*scale), int(h*scale)
                    small_image = cv2.resize(processed_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
                    h, w, c = small_image.shape
                    qimage = QImage(small_image.data, w, h, w * c, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimage)
                    self.update_image_signal.emit(pixmap)
                except Exception as e:
                    error_msg = f"Failed to display image: {str(e)}"
                    self.update_log_signal.emit(write_log_entry(self.paths['log_file'], error_msg))

                time.sleep(interval)
            except Exception as e:
                error_msg = f"Experiment error: {str(e)}"
                self.update_log_signal.emit(write_log_entry(self.paths['log_file'], error_msg))
                self.running = False
                break

        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_index_combo.setEnabled(True)
        self.folder_edit.setEnabled(True)
        self.interval_edit.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.hours_edit.setEnabled(True)
        self.minutes_edit.setEnabled(True)
        self.save_real_cb.setEnabled(True)
        self.save_processed_cb.setEnabled(True)
        if self.camera:
            self.camera.release()
            self.camera = None
        self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment stopped"))

    def stop_experiment(self):
        self.running = False
        self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment stopped by user"))

    def save_and_exit(self):
        self.running = False
        experiment_ran = False
        if self.paths.get('csv_file') and os.path.exists(self.paths['csv_file']) and os.path.getsize(self.paths['csv_file']) > 0:
            experiment_ran = True
            if self.coordinate_buffer:
                save_coordinates_to_csv_batch(self.coordinate_buffer, self.paths['csv_file'], 
                                            self.paths['log_file'], self.total_rows)
                self.coordinate_buffer = []

        if self.camera:
            self.camera.release()
            self.camera = None

        if experiment_ran:
            self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment saved and exited"))
            QMessageBox.information(self, "Success", "All files are successfully saved.")
        else:
            self.update_log_signal.emit("No experiment was run")
            QMessageBox.information(self, "Info", "No experiment was run.")

        self.close()
        QApplication.quit()

# === Main ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlyDetectionGUI()
    window.show()
    sys.exit(app.exec())