import sys
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QTextEdit, QFileDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime
import os

class DefectDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weld Defect Detection System")
        self.setGeometry(100, 100, 1000, 700)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Video display label with fixed size
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.video_label.setFixedSize(640, 480)
        self.main_layout.addWidget(self.video_label)

        # Status and defect labels layout
        self.info_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-size: 14px; color: #333;")
        self.defect_label = QLabel("Defect: None")
        self.defect_label.setStyleSheet("font-size: 14px; color: #333;")
        self.defect_type_label = QLabel("Defect Types: None")
        self.defect_type_label.setStyleSheet("font-size: 14px; color: #333;")
        self.info_layout.addWidget(self.status_label)
        self.info_layout.addWidget(self.defect_label)
        self.info_layout.addWidget(self.defect_type_label)
        self.main_layout.addLayout(self.info_layout)

        # Defect log
        self.defect_log = QTextEdit()
        self.defect_log.setReadOnly(True)
        self.defect_log.setStyleSheet("font-size: 12px; background-color: #f0f0f0; border: 1px solid #ccc;")
        self.defect_log.setFixedHeight(100)
        self.main_layout.addWidget(QLabel("Defect Log:"))
        self.main_layout.addWidget(self.defect_log)

        # Control buttons layout
        self.control_layout = QHBoxLayout()
        
        # Start/Stop button
        self.start_stop_button = QPushButton("Start Inspection")
        self.start_stop_button.setStyleSheet("""
            QPushButton {background-color: #4CAF50; color: white; padding: 8px; border-radius: 5px;}
            QPushButton:hover {background-color: #45a049;}
        """)
        self.start_stop_button.clicked.connect(self.toggle_inspection)
        self.control_layout.addWidget(self.start_stop_button)

        # Save snapshot button
        self.save_button = QPushButton("Save Snapshot")
        self.save_button.setStyleSheet("""
            QPushButton {background-color: #2196F3; color: white; padding: 8px; border-radius: 5px;}
            QPushButton:hover {background-color: #1e88e5;}
        """)
        self.save_button.clicked.connect(self.save_snapshot)
        self.control_layout.addWidget(self.save_button)

        # Test image button
        self.test_image_button = QPushButton("Test Image")
        self.test_image_button.setStyleSheet("""
            QPushButton {background-color: #FF9800; color: white; padding: 8px; border-radius: 5px;}
            QPushButton:hover {background-color: #fb8c00;}
        """)
        self.test_image_button.clicked.connect(self.test_image)
        self.control_layout.addWidget(self.test_image_button)

        # Camera selection
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Camera {i}" for i in range(3)])
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        self.camera_combo.setStyleSheet("padding: 5px;")
        self.control_layout.addWidget(QLabel("Select Camera:"))
        self.control_layout.addWidget(self.camera_combo)

        self.main_layout.addLayout(self.control_layout)

        # Initialize variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False
        self.current_camera = 0
        self.test_frame = None

        # Load models with error handling
        try:
            self.cnn_model = tf.keras.models.load_model(r"C:\Users\Sourav Kumar\OneDrive\Desktop\major project\fabric_defect_classifier.keras")
            self.defect_log.append("CNN model loaded successfully")
            self.defect_log.append(f"CNN input shape: {self.cnn_model.input_shape}")
        except Exception as e:
            self.defect_log.append(f"Error loading CNN model: {str(e)}")
            self.cnn_model = None

        try:
            self.yolo_model = YOLO(r"C:\Users\Sourav Kumar\OneDrive\Desktop\major project\FRONTEND\runs\detect\welding_defect2\weights\best.pt")
            self.defect_log.append("YOLO model loaded successfully")
            self.defect_log.append(f"YOLO classes: {self.yolo_model.names}")
        except Exception as e:
            self.defect_log.append(f"Error loading YOLO model: {str(e)}")
            self.yolo_model = None

        # Create snapshots directory
        self.snapshot_dir = "snapshots"
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)

    def toggle_inspection(self):
        if not self.is_running:
            self.start_inspection()
        else:
            self.stop_inspection()

    def start_inspection(self):
        if self.cnn_model is None or self.yolo_model is None:
            self.status_label.setText("Status: Error - Models not loaded")
            return

        self.cap = cv2.VideoCapture(self.current_camera)
        if not self.cap.isOpened():
            self.status_label.setText("Status: Error - Could not open camera")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.test_frame = None  # Reset test frame

        self.status_label.setText("Status: Inspecting...")
        self.start_stop_button.setText("Stop Inspection")
        self.start_stop_button.setStyleSheet("""
            QPushButton {background-color: #f44336; color: white; padding: 8px; border-radius: 5px;}
            QPushButton:hover {background-color: #e53935;}
        """)
        self.is_running = True
        self.timer.start(50)

    def stop_inspection(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.is_running = False
        self.test_frame = None
        self.start_stop_button.setText("Start Inspection")
        self.start_stop_button.setStyleSheet("""
            QPushButton {background-color: #4CAF50; color: white; padding: 8px; border-radius: 5px;}
            QPushButton:hover {background-color: #45a049;}
        """)
        self.status_label.setText("Status: Stopped")
        self.defect_label.setText("Defect: None")
        self.defect_type_label.setText("Defect Types: None")
        self.video_label.clear()

    def save_snapshot(self):
        if not self.is_running or (not self.cap and self.test_frame is None):
            self.status_label.setText("Status: No active inspection to save")
            return

        if self.test_frame is not None:
            frame = self.test_frame
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.setText("Status: Error - Could not read frame")
                return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.snapshot_dir, f"snapshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        self.defect_log.append(f"[{timestamp}] Snapshot saved: {filename}")

    def change_camera(self, index):
        if self.is_running:
            self.stop_inspection()
        self.current_camera = index
        self.status_label.setText(f"Status: Switched to Camera {index}")

    def test_image(self):
        if self.is_running:
            self.stop_inspection()

        file_name, _ = QFileDialog.getOpenFileName(self, "Select Test Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.test_frame = cv2.imread(file_name)
            if self.test_frame is None:
                self.status_label.setText("Status: Error - Could not load image")
                self.defect_log.append(f"Error loading test image: {file_name}")
                return
            self.status_label.setText("Status: Testing image...")
            self.is_running = True
            self.update_frame()

    def update_frame(self):
        if not self.is_running:
            return

        if self.test_frame is not None:
            frame = self.test_frame.copy()
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.setText("Status: Error - Could not read frame")
                self.stop_inspection()
                return

        # Enhance frame contrast
        frame_enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

        # Debug input statistics
        img_stats = f"Input stats: min={frame_enhanced.min()}, max={frame_enhanced.max()}, mean={frame_enhanced.mean():.2f}"
        print(img_stats)

        # Process frame for CNN
        cnn_input = cv2.resize(frame_enhanced, (224, 224))
        cnn_input_normalized = cnn_input / 255.0
        # Alternative normalization (zero-centering, if model expects it)
        # cnn_input_normalized = (cnn_input - 127.5) / 127.5
        cnn_input = np.expand_dims(cnn_input_normalized, axis=0)
        cnn_pred = self.cnn_model.predict(cnn_input, verbose=0)[0][0] if self.cnn_model else 0.0
        print(f"CNN prediction: {cnn_pred:.4f}")
        is_defective = cnn_pred > 0.5
        label = "Defect" if is_defective else "No Defect"
        color = (0, 0, 255) if is_defective else (0, 255, 0)

        # YOLOv11 Defect Type Detection
        defect_types = []
        all_detections = []
        if self.yolo_model:
            results = self.yolo_model(frame_enhanced)
            print(f"YOLO detections: {len(results[0].boxes)}")
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    defect_name = self.yolo_model.names[cls]
                    all_detections.append(f"{defect_name} ({conf:.2f})")
                    if conf > 0.3:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"{defect_name} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        defect_types.append(f"{defect_name} ({conf:.2f})")

        # Update defect labels and log
        self.defect_label.setText(f"Defect: {label} (Score: {cnn_pred:.4f})")
        defect_text = ', '.join(defect_types) if defect_types else f'None ({len(all_detections)} detections)'
        self.defect_type_label.setText(f"Defect Types: {defect_text}")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.defect_log.append(f"[{timestamp}] {img_stats}")
        self.defect_log.append(f"[{timestamp}] CNN Score: {cnn_pred:.4f}, YOLO: {defect_text}")
        if is_defective and defect_types:
            self.defect_log.append(f"[{timestamp}] Defect: {label}, Types: {defect_text}")
        if all_detections:
            self.defect_log.append(f"[{timestamp}] All YOLO detections: {', '.join(all_detections)}")

        # Add label to frame
        cv2.putText(frame, f"{label} (Score: {cnn_pred:.4f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Convert frame to QImage for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.stop_inspection()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DefectDetectionUI()
    window.show()
    sys.exit(app.exec_())