import time
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from core.detector import ObjectTracker


class VideoWorker(QThread):

    frame_ready = Signal(np.ndarray, float)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path
        self._running = True

    def run(self):
        tracker = ObjectTracker()
        cap = cv2.VideoCapture(self.video_path)

        while self._running and cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            detections = tracker.process_frame(frame)

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"#{det['id']} {det["class_name"]} ({det['confidence']})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2,
                )

            fps = 1.0 / max(time.time() - start_time, 1e-5)
            self.frame_ready.emit(frame, fps)

        cap.release()

    def stop(self):
        self._running = False
        self.wait()


class MainWindow(QMainWindow):

    def __init__(self, video_path: str):
        super().__init__()
        self.setWindowTitle("ADAS Core - Preview in RL")
        self.resize(960, 540)

        self.video_label = QLabel("Ładowanie modelu i wideo...")
        self.video_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = VideoWorker(video_path)
        self.worker.frame_ready.connect(self.update_image)
        self.worker.start()

    def update_image(self, frame: np.ndarray, fps: float):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qt_img).scaled(
                self.video_label.size(), Qt.KeepAspectRatio
            )
        )
        self.setWindowTitle(f"ADAS Core - FPS: {fps:.1f}")

    def close_event(self, event):
        self.worker.stop()
        super().closeEvent(event)
