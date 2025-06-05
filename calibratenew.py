import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QRadioButton,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QInputDialog, QMessageBox,
    QLabel, QComboBox, QSlider, QPushButton, QFileDialog
)
from PySide6.QtGui import QImage, QPixmap, QTransform, QCursor
from PySide6.QtCore import Qt, QPointF
import json
import logging
from dataclasses import dataclass
from collections import deque

# === CONFIG ===
PREVIEW_SCALE = 0.7
DEFAULT_CAMERA_INDEX = 0
CAMERA_RESOLUTIONS = [(1920, 1080), (1280, 720), (960, 720), (960, 540), (640, 480)]
ZOOM_STEP = 0.1
MIN_ZOOM = 0.5
MAX_ZOOM = 4.0
ARROW_STEP = 5
LINE_THICKNESS = 3
SELECTED_LINE_THICKNESS = 7
DUPLICATE_OFFSET = 10  # Offset for duplicated column

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === ColumnConfig Dataclass ===
@dataclass
class ColumnConfig:
    left: int
    right: int
    top: int
    bottom: int
    rows: int
    v_gap: int = 0

    def to_dict(self):
        return {
            'left': self.left,
            'right': self.right,
            'top': self.top,
            'bottom': self.bottom,
            'rows': self.rows,
            'v_gap': self.v_gap
        }

    @staticmethod
    def from_dict(d):
        return ColumnConfig(**d)

    def get_points(self):
        return [
            QPointF(self.left, self.top),    # 0: top-left
            QPointF(self.right, self.top),   # 1: top-right
            QPointF(self.right, self.bottom),# 2: bottom-right
            QPointF(self.left, self.bottom)  # 3: bottom-left
        ]

    def update_from_points(self, points):
        x_coords = [p.x() for p in points]
        y_coords = [p.y() for p in points]
        self.left = int(min(x_coords))
        self.right = int(max(x_coords))
        self.top = int(min(y_coords))
        self.bottom = int(max(y_coords))

    def transform(self, transform_matrix):
        points = self.get_points()
        points_array = np.float32([[p.x(), p.y()] for p in points]).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points_array, transform_matrix)
        transformed_points = [QPointF(pt[0][0], pt[0][1]) for pt in transformed_points]
        self.update_from_points(transformed_points)

# === Zoomable Graphics View ===
class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.zoom_level = 1.0
        self.pixmap_item = None
        self.initialized = False
        self.parent = parent
        self.mode = 'drag'
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.dot_points = []
        self.warp_points = []  # List to store points for warp perspective
        self.selected_column = None
        self.selected_point = None
        self.dragging = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.NoFocus)

    def set_pixmap(self, pixmap):
        if self.pixmap_item:
            self.scene().removeItem(self.pixmap_item)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        if not self.initialized:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
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
                cursor_pos = event.position().toPoint()
                scene_pos_before = self.mapToScene(cursor_pos)
                self.zoom_level = new_zoom
                self.resetTransform()
                self.scale(self.zoom_level, self.zoom_level)
                scene_pos_after = self.mapToScene(cursor_pos)
                delta_pos = scene_pos_after - scene_pos_before
                self.translate(delta_pos.x(), delta_pos.y())
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            img_pos = self.scene_to_image_pos(pos)
            if not self.is_within_image(img_pos):
                return

            if self.mode == 'warp':
                self.warp_points.append(img_pos)
                self.setCursor(Qt.ArrowCursor)
                if len(self.warp_points) == 4:
                    self.parent.done_btn.setEnabled(True)
                self.parent.update_display()
                return

            for idx, col in enumerate(self.parent.columns):
                points = col.get_points()
                for i, pt in enumerate(points):
                    if ((pt.x() - img_pos.x())**2 + (pt.y() - img_pos.y())**2)**0.5 < 10:
                        self.selected_column = idx
                        self.selected_point = i
                        self.dragging = True
                        self.setCursor(Qt.SizeAllCursor)
                        self.parent.push_undo()
                        self.parent.update_display()
                        return
                if (col.left <= img_pos.x() <= col.right and
                    col.top <= img_pos.y() <= col.bottom):
                    self.parent.selected_column_idx = idx + 1
                    self.parent.update_display()
                    return

            if self.mode == 'drag':
                self.drawing = True
                self.start_point = img_pos
                self.current_point = img_pos
                self.setCursor(Qt.CrossCursor)
            elif self.mode == 'dot':
                self.dot_points.append(img_pos)
                self.setCursor(Qt.CrossCursor)
                if len(self.dot_points) == 4:
                    self.create_column_from_points()
                self.parent.update_display()

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        img_pos = self.scene_to_image_pos(pos)
        if self.is_within_image(img_pos):
            if self.dragging and self.selected_column is not None and self.selected_point is not None: # This is when resizing an existing column
                self.setCursor(Qt.CrossCursor)
                col = self.parent.columns[self.selected_column]
                new_x = int(img_pos.x())
                new_y = int(img_pos.y())

                # self.selected_point is 0:TL, 1:TR, 2:BR, 3:BL
                if self.selected_point == 0:  # Top-left
                    col.left = new_x
                    col.top = new_y
                elif self.selected_point == 1:  # Top-right
                    col.right = new_x
                    col.top = new_y
                elif self.selected_point == 2:  # Bottom-right
                    col.right = new_x
                    col.bottom = new_y
                elif self.selected_point == 3:  # Bottom-left
                    col.left = new_x
                    col.bottom = new_y

                # Ensure that left < right and top < bottom.
                # If dragging, e.g., the left edge past the right edge, they swap.
                if col.left > col.right:
                    col.left, col.right = col.right, col.left

                if col.top > col.bottom:
                    col.top, col.bottom = col.bottom, col.top

                self.parent.update_display()
            elif self.mode == 'drag' and self.drawing:
                self.setCursor(Qt.CrossCursor)
                self.current_point = img_pos
                self.parent.update_display()
            elif self.mode == 'drag' and not self.dragging:
                self.setCursor(Qt.CrossCursor)
            elif self.mode == 'dot':
                self.setCursor(Qt.CrossCursor)
            elif self.mode == 'warp':
                self.setCursor(Qt.ArrowCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            try:
                if self.drawing and self.mode == 'drag':
                    self.drawing = False
                    if self.start_point and self.current_point:
                        left = min(self.start_point.x(), self.current_point.x())
                        right = max(self.start_point.x(), self.current_point.x())
                        top = min(self.start_point.y(), self.current_point.y())
                        bottom = max(self.start_point.y(), self.current_point.y())
                        if left != right and top != bottom:
                            self.parent.processing_dialog = True
                            rows, ok = QInputDialog.getInt(
                                self, "Set Rows", "Number of table rows:", 5, 1, 100, 1
                            )
                            self.parent.processing_dialog = False
                            if ok:
                                self.parent.push_undo()
                                col = ColumnConfig(left=left, right=right, top=top, bottom=bottom, rows=rows)
                                self.parent.columns.append(col)
                                self.parent.selected_column_idx = len(self.parent.columns)
                                self.parent.update_display()
                        self.start_point = None
                        self.current_point = None
                    self.parent.update_display()
                elif self.dragging:
                    self.dragging = False
                    self.selected_column = None
                    self.selected_point = None
                    self.unsetCursor()
                    self.parent.update_display()
            except Exception as e:
                logging.error(f"Error in mouseReleaseEvent: {e}")
                self.drawing = False
                self.start_point = None
                self.current_point = None
                self.parent.update_display()

    def create_column_from_points(self):
        if len(self.dot_points) == 4:
            self.parent.processing_dialog = True
            try:
                rows, ok = QInputDialog.getInt(
                    self, "Set Rows", "Number of table rows:", 5, 1, 100, 1
                )
                self.parent.processing_dialog = False # Moved this up to be set regardless
                if not ok: # User cancelled
                    self.dot_points = []
                    self.parent.update_display()
                    return

                # if ok: # This 'if ok:' is now redundant due to the early return
                self.parent.push_undo()
                points = self.order_points_clockwise()
                left = int(min(p.x() for p in points))
                right = int(max(p.x() for p in points))
                top = int(min(p.y() for p in points))
                bottom = int(max(p.y() for p in points))
                if left != right and top != bottom:
                        col = ColumnConfig(left=left, right=right, top=top, bottom=bottom, rows=rows)
                        self.parent.columns.append(col)
                        self.parent.selected_column_idx = len(self.parent.columns)
                else:
                        QMessageBox.warning(self, "Invalid Rectangle", "The points do not form a valid rectangle.")
                self.dot_points = []
                self.parent.update_display()
            except Exception as e:
                logging.error(f"Error creating column from points: {e}")
                self.parent.processing_dialog = False
                self.dot_points = []
                self.parent.update_display()

    def order_points_clockwise(self):
        points = self.dot_points
        cx = sum(p.x() for p in points) / 4
        cy = sum(p.y() for p in points) / 4
        sorted_points = sorted(points, key=lambda p: -np.arctan2(p.y() - cy, p.x() - cx))
        top_left = min(range(4), key=lambda i: sorted_points[i].x() + sorted_points[i].y())
        return sorted_points[top_left:] + sorted_points[:top_left]

    def scene_to_image_pos(self, scene_pos):
        if self.pixmap_item:
            pixmap_rect = self.pixmap_item.boundingRect()
            img_width, img_height = self.parent.camera_width, self.parent.camera_height
            scale_x = img_width / pixmap_rect.width()
            scale_y = img_height / pixmap_rect.height()
            img_x = int((scene_pos.x() - pixmap_rect.left()) * scale_x)
            img_y = int((scene_pos.y() - pixmap_rect.top()) * scale_y)
            return QPointF(img_x, img_y)
        return scene_pos

    def is_within_image(self, img_pos):
        return (0 <= img_pos.x() <= self.parent.camera_width and
                0 <= img_pos.y() <= self.parent.camera_height)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            self.parent.keyPressEvent(event)
        else:
            super().keyPressEvent(event)

# === Main Application ===
class CalibrationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibration Tool")
        self.resize(int(self.screen().size().width() * 1), int(self.screen().size().height() * 1))
        self.columns = []
        self.cap = None
        self.current_image = None
        self.original_image = None
        self.camera_width = 1920
        self.camera_height = 1080
        self.selected_column_idx = None
        self.undo_stack = deque(maxlen=50)
        self.redo_stack = deque(maxlen=50)
        self.processing_dialog = False
        self.warp_transforms = []
        self.last_warp_points_original = []
        self.init_ui()
        self.init_camera()
        self.setFocusPolicy(Qt.StrongFocus)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        control_widget = QWidget()
        control_widget.setMaximumWidth(min(350, int(self.screen().size().width() * 0.25)))
        control_widget.setMinimumWidth(250)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(8, 8, 8, 8)

        self.setStyleSheet("""
            QWidget {
                background-color: #2E2E2E;
                color: #FFFFFF;
            }
            QPushButton {
                padding: 8px;
                font-size: 14px;
                border-radius: 5px;
                background-color: #4A4A4A;
                border: 1px solid #666666;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #3A3A3A;
                color: #666666;
            }
            QPushButton:hover {
                background-color: #5A5A5A;
            }
            QPushButton:pressed {
                background-color: #3A3A3A;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 5px solid #666666;
                border-radius: 5px;
                margin-top: 10px;
                color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                color: #FFFFFF;
            }
            QLabel {
                font-size: 12px;
                color: #FFFFFF;
            }
            QRadioButton, QComboBox, QSlider {
                font-size: 12px;
                color: #FFFFFF;
                background-color: #3A3A3A;
                border: 1px solid #666666;
            }
            QRadioButton::indicator:checked {
                background-color: #4A90E2;
            }
            QComboBox, QSlider {
                padding: 2px;
            }
        """)

        self.preview_view = ZoomableGraphicsView(self)
        main_layout.addWidget(control_widget, 1)
        main_layout.addWidget(self.preview_view, 3)

        action_layout = QHBoxLayout()
        self.undo_btn = QPushButton("Undo (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn = QPushButton("Redo (Ctrl+Y)")
        self.redo_btn.clicked.connect(self.redo)
        self.exit_btn = QPushButton("Exit (Esc)")
        self.exit_btn.setStyleSheet("background-color: #D32F2F; border: 1px solid #B71C1C;")
        self.exit_btn.clicked.connect(self.close)
        action_layout.addWidget(self.undo_btn)
        action_layout.addWidget(self.redo_btn)
        action_layout.addWidget(self.exit_btn)
        control_layout.addLayout(action_layout)

        save_load_layout = QHBoxLayout()
        save_btn = QPushButton("Save ROI")
        save_btn.clicked.connect(self.save_layout)
        load_btn = QPushButton("Load ROI")
        load_btn.clicked.connect(self.load_layout)
        save_load_layout.addWidget(save_btn)
        save_load_layout.addWidget(load_btn)
        control_layout.addLayout(save_load_layout)

        capture_layout = QHBoxLayout()
        capture_btn = QPushButton("Capture Image")
        capture_btn.clicked.connect(self.capture_image)
        capture_layout.addWidget(capture_btn)
        control_layout.addLayout(capture_layout)

        column_action_layout = QHBoxLayout()
        delete_btn = QPushButton("Delete Column")
        delete_btn.clicked.connect(self.delete_column)
        delete_btn.setStyleSheet("background-color: #D32F2F; border: 1px solid #B71C1C;")
        duplicate_btn = QPushButton("Duplicate Column")
        duplicate_btn.clicked.connect(self.duplicate_column)
        edit_rows_btn = QPushButton("Edit Rows")
        edit_rows_btn.clicked.connect(self.edit_rows)
        column_action_layout.addWidget(delete_btn)
        column_action_layout.addWidget(duplicate_btn)
        column_action_layout.addWidget(edit_rows_btn)
        control_layout.addLayout(column_action_layout)

        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.drag_mode = QRadioButton("Drag Mode")
        self.drag_mode.setChecked(True)
        self.dot_mode = QRadioButton("Dot Mode")
        self.drag_mode.toggled.connect(lambda: self.set_mode('drag'))
        self.dot_mode.toggled.connect(lambda: self.set_mode('dot'))
        mode_layout.addWidget(self.drag_mode)
        mode_layout.addWidget(self.dot_mode)
        control_layout.addWidget(mode_group)

        warp_group = QGroupBox("Warp Perspective")
        warp_layout = QVBoxLayout(warp_group)
        self.warp_mode = QRadioButton("Warp Mode")
        self.warp_mode.toggled.connect(lambda: self.set_mode('warp'))
        warp_layout.addWidget(self.warp_mode)
        self.done_btn = QPushButton("Done")
        self.done_btn.setEnabled(False)
        self.done_btn.clicked.connect(self.warp_perspective)
        warp_layout.addWidget(self.done_btn)
        control_layout.addWidget(warp_group)

        camera_group = QGroupBox("Camera")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.addWidget(QLabel("Camera Index:"))
        self.camera_index_selector = QComboBox()
        self.camera_index_selector.addItems(["0", "1", "2", "3"])
        self.camera_index_selector.setCurrentIndex(DEFAULT_CAMERA_INDEX)
        self.camera_index_selector.currentIndexChanged.connect(self.update_camera)
        camera_layout.addWidget(self.camera_index_selector)
        camera_layout.addWidget(QLabel("Exposure:"))
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setMinimum(-13)
        self.exposure_slider.setMaximum(-1)
        self.exposure_slider.setValue(-13)
        self.exposure_slider.valueChanged.connect(self.adjust_exposure)
        camera_layout.addWidget(self.exposure_slider)
        control_layout.addWidget(camera_group)

        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        instructions_layout.addWidget(QLabel("- Click 'Capture Image' to start"))
        instructions_layout.addWidget(QLabel("- Drag Mode: Click and drag to draw"))
        instructions_layout.addWidget(QLabel("- Dot Mode: Click 4 points to form rectangle"))
        instructions_layout.addWidget(QLabel("- Warp Mode: Click 4 points, then click 'Done'"))
        instructions_layout.addWidget(QLabel("- Click red point to resize column"))
        instructions_layout.addWidget(QLabel("- Click column to select"))
        instructions_layout.addWidget(QLabel("- Arrow keys to move, Shift+Arrow to resize"))
        instructions_layout.addWidget(QLabel("- Use 'Duplicate/Delete Column' or 'Edit Rows'"))
        instructions_layout.addWidget(QLabel("- Ctrl+Wheel to zoom, drag to pan"))
        control_layout.addWidget(instructions_group)

        control_layout.addStretch()

    def set_mode(self, mode):
        self.preview_view.mode = mode
        if mode == 'dot':
            self.preview_view.dot_points = []
        elif mode == 'warp':
            self.preview_view.warp_points = []
            self.done_btn.setEnabled(False)
        self.update_display()

    def init_camera(self):
        self.update_camera(DEFAULT_CAMERA_INDEX)
        self.adjust_exposure(-13)

    def update_camera(self, index):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            logging.error(f"Failed to open camera at index {index}")
            QMessageBox.critical(self, "Error", f"Failed to open camera at index {index}")
            self.cap = None
            return
        for width, height in CAMERA_RESOLUTIONS:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.camera_width >= 640 and self.camera_height >= 480:
                break

    def capture_image(self):
        if self.cap is None or not self.cap.isOpened():
            self.update_camera(self.camera_index_selector.currentIndex())
        
        if self.cap is None or not self.cap.isOpened():
            logging.error("No valid camera to capture image")
            QMessageBox.critical(self, "Error", "No valid camera. Please select a working camera index.")
            return
        try:
            ret, frame = self.cap.read()
            if ret:
                self.original_image = frame.copy()
                cumulative_transform = self.get_cumulative_transform()
                self.current_image = cv2.warpPerspective(self.original_image, cumulative_transform, (self.camera_width, self.camera_height))
                self.preview_view.warp_points = []
                self.done_btn.setEnabled(False)
                self.warp_mode.setChecked(False)
                self.drag_mode.setChecked(True)
                self.set_mode('drag')
                self.update_display()
            else:
                logging.error("Failed to capture image")
                QMessageBox.critical(self, "Error", "Failed to capture image from camera.")
        except Exception as e:
            logging.error(f"Error capturing image: {e}")
            QMessageBox.critical(self, "Error", f"Image capture failed: {e}")

    def adjust_exposure(self, value):
        if self.cap is None or not self.cap.isOpened():
            return
        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        except Exception as e:
            logging.error(f"Failed to set exposure: {e}")
            if not hasattr(self.adjust_exposure, 'warned'):
                QMessageBox.warning(self, "Warning", "Exposure adjustment not supported by this camera.")
                self.adjust_exposure.warned = True

    def get_cumulative_transform(self, up_to_index=None):
        if not self.warp_transforms:
            return np.eye(3, dtype=np.float32)
        transforms = self.warp_transforms[:up_to_index] if up_to_index is not None else self.warp_transforms
        if not transforms:
            return np.eye(3, dtype=np.float32)
        cumulative = np.eye(3, dtype=np.float32)
        for transform in transforms:
            cumulative = np.dot(transform, cumulative)
        return cumulative

    def warp_perspective(self):
        if len(self.preview_view.warp_points) != 4:
            QMessageBox.warning(self, "Invalid Points", "Please select exactly 4 points in Warp Mode.")
            return

        try:
            self.push_undo()
            src_points = np.float32([[p.x(), p.y()] for p in self.preview_view.warp_points])
            dst_points = np.float32([
                [0, 0],
                [self.camera_width - 1, 0],
                [self.camera_width - 1, self.camera_height - 1],
                [0, self.camera_height - 1]
            ])
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            if self.warp_transforms:
                inverse_transform = cv2.invert(self.get_cumulative_transform())[1]
                points_array = np.float32([[p.x(), p.y()] for p in self.preview_view.warp_points]).reshape(-1, 1, 2)
                transformed_points = cv2.perspectiveTransform(points_array, inverse_transform)
                self.last_warp_points_original = [QPointF(pt[0][0], pt[0][1]) for pt in transformed_points]
            else:
                self.last_warp_points_original = [QPointF(p.x(), p.y()) for p in self.preview_view.warp_points]
            self.warp_transforms.append(transform_matrix)
            self.current_image = cv2.warpPerspective(
                self.original_image, self.get_cumulative_transform(), (self.camera_width, self.camera_height)
            )
            self.preview_view.warp_points = []
            self.done_btn.setEnabled(False)
            self.warp_mode.setChecked(False)
            self.drag_mode.setChecked(True)
            self.set_mode('drag')
            self.update_display()
        except Exception as e:
            logging.error(f"Error applying perspective transform: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply perspective transform: {e}")

    def update_display(self):
        if self.processing_dialog:
            return
        if self.current_image is None:
            return
        try:
            temp_rect = None
            dot_points = self.preview_view.dot_points if self.preview_view.mode == 'dot' else None
            warp_points = self.preview_view.warp_points if self.preview_view.mode == 'warp' else None
            if self.preview_view.drawing and self.preview_view.start_point and self.preview_view.current_point:
                temp_rect = (self.preview_view.start_point, self.preview_view.current_point)
            preview = self.draw_columns(self.current_image, self.selected_column_idx, temp_rect, dot_points, warp_points)
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            height, width, _ = preview_rgb.shape
            qimage = QImage(preview_rgb.data, width, height, width * 3, QImage.Format_RGB888)
            max_width = int(self.screen().size().width() * PREVIEW_SCALE)
            max_height = int(self.screen().size().height() * 0.9)
            pixmap = QPixmap.fromImage(qimage).scaled(max_width, max_height, Qt.KeepAspectRatio)
            self.preview_view.set_pixmap(pixmap)
        except Exception as e:
            logging.error(f"Error updating display: {e}")
            QMessageBox.critical(self, "Error", f"Display update failed: {e}")

    def draw_columns(self, image, selected_idx=None, temp_rect=None, dot_points=None, warp_points=None):
        preview = image.copy()
        if selected_idx is not None:
            idx = selected_idx - 1
            if 0 <= idx < len(self.columns):
                col = self.columns[idx]
                if col.rows > 0 and col.left < col.right and col.top < col.bottom:
                    roi_height = (col.bottom - col.top - (col.rows - 1) * col.v_gap) // col.rows
                    if roi_height > 0:
                        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                        for row in range(col.rows):
                            x1 = max(0, col.left)
                            y1 = max(0, col.top + row * (roi_height + col.v_gap))
                            x2 = min(image.shape[1], col.right)
                            y2 = min(image.shape[0], y1 + roi_height)
                            if x1 < x2 and y1 < y2:
                                cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, LINE_THICKNESS)
                        kernel = np.ones((3, 3), np.uint8)
                        glow_mask = cv2.dilate(mask, kernel, iterations=1)
                        glow_mask = cv2.subtract(glow_mask, mask)
                        glow_indices = glow_mask > 0
                        preview[glow_indices] = preview[glow_indices] * 0.7 + np.array([0, 255, 255]) * 0.3

        for idx, col in enumerate(self.columns, start=1):
            if col.rows <= 0 or col.left >= col.right or col.top >= col.bottom:
                continue
            roi_height = (col.bottom - col.top - (col.rows - 1) * col.v_gap) // col.rows
            if roi_height <= 0:
                continue
            color = (255, 0, 0) if idx == selected_idx else (0, 255, 0)
            thickness = SELECTED_LINE_THICKNESS if idx == selected_idx else LINE_THICKNESS
            for row in range(col.rows):
                x1 = max(0, col.left)
                y1 = max(0, col.top + row * (roi_height + col.v_gap))
                x2 = min(image.shape[1], col.right)
                y2 = min(image.shape[0], y1 + roi_height)
                if x1 >= x2 or y1 >= y2:
                    continue
                cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                cv2.putText(preview, f'C{idx}R{row+1}', (int(x1 + 5), int(y1 + 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for pt in col.get_points():
                x, y = int(pt.x()), int(pt.y())
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)
        if temp_rect:
            try:
                x1, y1 = int(temp_rect[0].x()), int(temp_rect[0].y())
                x2, y2 = int(temp_rect[1].x()), int(temp_rect[1].y())
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if x1 != x2 and y1 != y2 and 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and 0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]:
                    cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), LINE_THICKNESS)
            except Exception as e:
                logging.error(f"Error drawing temp rect: {e}")
        if warp_points:
            for i, pt in enumerate(warp_points):
                x, y = int(pt.x()), int(pt.y())
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(preview, (x, y), 5, (255, 0, 255), -1)
                    if i > 0:
                        cv2.line(preview, (int(warp_points[i-1].x()), int(warp_points[i-1].y())),
                                (x, y), (255, 0, 255), LINE_THICKNESS)

        if dot_points: # Draw dot_points if they exist
            for i, pt in enumerate(dot_points):
                x, y = int(pt.x()), int(pt.y())
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(preview, (x, y), 5, (0, 255, 255), -1) # Yellow circle
                    if i > 0:
                        prev_x, prev_y = int(dot_points[i-1].x()), int(dot_points[i-1].y())
                        if 0 <= prev_x < image.shape[1] and 0 <= prev_y < image.shape[0]:
                             cv2.line(preview, (prev_x, prev_y), (x,y), (0, 255, 255), LINE_THICKNESS)
            if len(dot_points) == 4: # If 4 points, close the polygon
                p3_x, p3_y = int(dot_points[3].x()), int(dot_points[3].y())
                p0_x, p0_y = int(dot_points[0].x()), int(dot_points[0].y())
                if (0 <= p3_x < image.shape[1] and 0 <= p3_y < image.shape[0] and
                    0 <= p0_x < image.shape[1] and 0 <= p0_y < image.shape[0]):
                    cv2.line(preview, (p3_x, p3_y), (p0_x, p0_y), (0, 255, 255), LINE_THICKNESS)
        return preview

    def delete_column(self):
        if self.selected_column_idx is None:
            QMessageBox.warning(self, "No Selection", "Please select a column to delete.")
            return
        idx = self.selected_column_idx - 1
        if 0 <= idx < len(self.columns):
            self.push_undo()
            self.columns.pop(idx)
            self.selected_column_idx = None
            self.update_display()
        else:
            QMessageBox.warning(self, "Invalid Selection", "Selected column index is invalid.")

    def duplicate_column(self):
        if self.selected_column_idx is None:
            QMessageBox.warning(self, "No Selection", "Please select a column to duplicate.")
            return
        idx = self.selected_column_idx - 1
        if 0 <= idx < len(self.columns):
            self.push_undo()
            col = self.columns[idx]
            new_col = ColumnConfig(
                left=col.left + DUPLICATE_OFFSET,
                right=col.right + DUPLICATE_OFFSET,
                top=col.top + DUPLICATE_OFFSET,
                bottom=col.bottom + DUPLICATE_OFFSET,
                rows=col.rows,
                v_gap=col.v_gap
            )
            # Ensure the new column stays within image bounds
            new_col.left = max(0, new_col.left)
            new_col.right = min(self.camera_width, new_col.right)
            new_col.top = max(0, new_col.top)
            new_col.bottom = min(self.camera_height, new_col.bottom)
            self.columns.append(new_col)
            self.selected_column_idx = len(self.columns)
            self.update_display()
        else:
            QMessageBox.warning(self, "Invalid Selection", "Selected column index is invalid.")

    def edit_rows(self):
        if self.selected_column_idx is None:
            QMessageBox.warning(self, "No Selection", "Please select a column to edit rows.")
            return
        idx = self.selected_column_idx - 1
        if 0 <= idx < len(self.columns):
            col = self.columns[idx]
            self.processing_dialog = True
            rows, ok = QInputDialog.getInt(
                self, "Edit Rows", "Number of table rows:", col.rows, 1, 100, 1
            )
            self.processing_dialog = False
            if ok:
                self.push_undo()
                self.columns[idx].rows = rows
                self.update_display()
        else:
            QMessageBox.warning(self, "Invalid Selection", "Selected column index is invalid.")

    def save_layout(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save ROI", "", "JSON Files (*.json)")
        if file_path:
            try:
                warp_points_data = []
                if self.preview_view.warp_points and len(self.preview_view.warp_points) == 4:
                    inverse_transform = cv2.invert(self.get_cumulative_transform())[1]
                    points_array = np.float32([[p.x(), p.y()] for p in self.preview_view.warp_points]).reshape(-1, 1, 2)
                    transformed_points = cv2.perspectiveTransform(points_array, inverse_transform)
                    warp_points_data = [{'x': pt[0][0], 'y': pt[0][1]} for pt in transformed_points]
                else:
                    warp_points_data = [{'x': p.x(), 'y': p.y()} for p in self.last_warp_points_original]
                roi_data = [col.to_dict() for col in self.columns]
                data = {
                    'warp_points': warp_points_data,
                    'rois': roi_data
                }
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                with open(file_path, 'r') as f:
                    loaded_data = json.load(f)
                if loaded_data != data:
                    raise ValueError("Saved JSON does not match the original data")
                QMessageBox.information(self, "Saved", "ROI saved successfully!")
            except Exception as e:
                logging.error(f"Failed to save layout: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def load_layout(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load ROI", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.push_undo()
                warp_points_data = data.get('warp_points', [])
                self.preview_view.warp_points = [QPointF(p['x'], p['y']) for p in warp_points_data]
                self.last_warp_points_original = self.preview_view.warp_points.copy()
                if len(self.preview_view.warp_points) == 4:
                    src_points = np.float32([[p.x(), p.y()] for p in self.preview_view.warp_points])
                    dst_points = np.float32([
                        [0, 0],
                        [self.camera_width - 1, 0],
                        [self.camera_width - 1, self.camera_height - 1],
                        [0, self.camera_height - 1]
                    ])
                    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    self.warp_transforms = [transform_matrix]
                    self.current_image = cv2.warpPerspective(
                        self.original_image, transform_matrix, (self.camera_width, self.camera_height)
                    )
                    self.preview_view.warp_points = []
                else:
                    self.warp_transforms = []
                    self.current_image = self.original_image.copy() if self.original_image is not None else None
                self.columns = [ColumnConfig.from_dict(d) for d in data.get('rois', [])]
                self.selected_column_idx = None
                self.done_btn.setEnabled(False)
                self.warp_mode.setChecked(False)
                self.drag_mode.setChecked(True)
                self.set_mode('drag')
                self.update_display()
                QMessageBox.information(self, "Loaded", "ROI loaded successfully!")
            except Exception as e:
                logging.error(f"Failed to load layout: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def push_undo(self):
        state = {
            'original_image': self.original_image.copy() if self.original_image is not None else None,
            'current_image': self.current_image.copy() if self.current_image is not None else None,
            'warp_transforms': [m.copy() for m in self.warp_transforms] if self.warp_transforms else [],
            'warp_points': [(p.x(), p.y()) for p in self.preview_view.warp_points],
            'last_warp_points_original': [(p.x(), p.y()) for p in self.last_warp_points_original],
            'columns': [ColumnConfig(
                left=col.left, right=col.right, top=col.top, bottom=col.bottom,
                rows=col.rows, v_gap=col.v_gap) for col in self.columns],
            'selected_column_idx': self.selected_column_idx
        }
        self.undo_stack.append(state)
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            current_state = {
                'original_image': self.original_image.copy() if self.original_image is not None else None,
                'current_image': self.current_image.copy() if self.current_image is not None else None,
                'warp_transforms': [m.copy() for m in self.warp_transforms] if self.warp_transforms else [],
                'warp_points': [(p.x(), p.y()) for p in self.preview_view.warp_points],
                'last_warp_points_original': [(p.x(), p.y()) for p in self.last_warp_points_original],
                'columns': [ColumnConfig(
                    left=col.left, right=col.right, top=col.top, bottom=col.bottom,
                    rows=col.rows, v_gap=col.v_gap) for col in self.columns],
                'selected_column_idx': self.selected_column_idx
            }
            self.redo_stack.append(current_state)
            previous_state = self.undo_stack.pop()
            self.original_image = previous_state['original_image'].copy() if previous_state['original_image'] is not None else None
            self.current_image = previous_state['current_image'].copy() if previous_state['current_image'] is not None else None
            self.warp_transforms = [m.copy() for m in previous_state['warp_transforms']] if previous_state['warp_transforms'] else []
            self.preview_view.warp_points = [QPointF(x, y) for x, y in previous_state['warp_points']]
            self.last_warp_points_original = [QPointF(x, y) for x, y in previous_state['last_warp_points_original']]
            self.columns = previous_state['columns']
            self.selected_column_idx = previous_state['selected_column_idx']
            self.done_btn.setEnabled(len(self.preview_view.warp_points) == 4)
            self.update_display()

    def redo(self):
        if self.redo_stack:
            current_state = {
                'original_image': self.original_image.copy() if self.original_image is not None else None,
                'current_image': self.current_image.copy() if self.current_image is not None else None,
                'warp_transforms': [m.copy() for m in self.warp_transforms] if self.warp_transforms else [],
                'warp_points': [(p.x(), p.y()) for p in self.preview_view.warp_points],
                'last_warp_points_original': [(p.x(), p.y()) for p in self.last_warp_points_original],
                'columns': [ColumnConfig(
                    left=col.left, right=col.right, top=col.top, bottom=col.bottom,
                    rows=col.rows, v_gap=col.v_gap) for col in self.columns],
                'selected_column_idx': self.selected_column_idx
            }
            self.undo_stack.append(current_state)
            next_state = self.redo_stack.pop()
            self.original_image = next_state['original_image'].copy() if next_state['original_image'] is not None else None
            self.current_image = next_state['current_image'].copy() if next_state['current_image'] is not None else None
            self.warp_transforms = [m.copy() for m in next_state['warp_transforms']] if next_state['warp_transforms'] else []
            self.preview_view.warp_points = [QPointF(x, y) for x, y in next_state['warp_points']]
            self.last_warp_points_original = [QPointF(x, y) for x, y in next_state['last_warp_points_original']]
            self.columns = next_state['columns']
            self.selected_column_idx = next_state['selected_column_idx']
            self.done_btn.setEnabled(len(self.preview_view.warp_points) == 4)
            self.update_display()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.undo()
        elif event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Y:
            self.redo()
        elif self.selected_column_idx is not None:
            idx = self.selected_column_idx - 1
            if idx < len(self.columns):
                self.push_undo()
                col = self.columns[idx]
                if event.modifiers() & Qt.ShiftModifier:
                    if event.key() == Qt.Key_Up:
                        col.top = max(0, col.top - ARROW_STEP)
                    elif event.key() == Qt.Key_Down:
                        col.bottom = min(self.camera_height, col.bottom + ARROW_STEP)
                    elif event.key() == Qt.Key_Left:
                        col.left = max(0, col.left - ARROW_STEP)
                    elif event.key() == Qt.Key_Right:
                        col.right = min(self.camera_width, col.right + ARROW_STEP)
                    if col.left >= col.right:
                        col.left = col.right - 1
                    if col.top >= col.bottom:
                        col.top = col.bottom - 1
                else:
                    dx = 0
                    dy = 0
                    if event.key() == Qt.Key_Up:
                        dy = -ARROW_STEP
                    elif event.key() == Qt.Key_Down:
                        dy = ARROW_STEP
                    elif event.key() == Qt.Key_Left:
                        dx = -ARROW_STEP
                    elif event.key() == Qt.Key_Right:
                        dx = ARROW_STEP
                    new_left = col.left + dx
                    new_right = col.right + dx
                    new_top = col.top + dy
                    new_bottom = col.bottom + dy
                    if new_left >= 0 and new_right <= self.camera_width and new_top >= 0 and new_bottom <= self.camera_height:
                        col.left = new_left
                        col.right = new_right
                        col.top = new_top
                        col.bottom = new_bottom
                    else:
                        if new_left < 0:
                            col.left = 0
                            col.right = col.left + (col.right - col.left)
                        elif new_right > self.camera_width:
                            col.right = self.camera_width
                            col.left = col.right - (col.right - col.left)
                        if new_top < 0:
                            col.top = 0
                            col.bottom = col.top + (col.bottom - col.top)
                        elif new_bottom > self.camera_height:
                            col.bottom = self.camera_height
                            col.top = col.bottom - (col.bottom - col.top)
                self.update_display()
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.selected_column_idx is not None:
            idx = self.selected_column_idx - 1
            if idx < len(self.columns):
                col = self.columns[idx]
                if col.left >= col.right or col.top >= col.bottom:
                    QMessageBox.warning(self, "Invalid Dimensions",
                        "Column dimensions are invalid (left >= right or top >= bottom). Adjust using arrow keys or resize.")
                self.processing_dialog = True
                rows, ok = QInputDialog.getInt(
                    self, "Edit Rows", "Number of table rows:", self.columns[idx].rows, 1, 100, 1
                )
                self.processing_dialog = False
                if ok:
                    self.push_undo()
                    self.columns[idx].rows = rows
                    self.update_display()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    window = CalibrationTool()
    window.show()
    sys.exit(app.exec())