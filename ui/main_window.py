import time
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from core.detector import ObjectTracker
from core.geometry import BirdEyeTransformer
from core.adas_logic import CollisionWarningSystem


class VideoWorker(QThread):

    frame_ready = Signal(np.ndarray, float)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path
        self._running = True

    def run(self):
        tracker = ObjectTracker()
        bev_transformer = BirdEyeTransformer()
        fcw_system = CollisionWarningSystem(20)
        cap = cv2.VideoCapture(self.video_path)

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        dt = 1.0 / video_fps

        while self._running and cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            detections = tracker.process_frame(frame)
            bev_view = bev_transformer.warp_to_bev(frame)

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]

                bottom_center = ((x1 + x2) / 2.0, float(y2))
                bev_x, bev_y = bev_transformer.transform_point(bottom_center)

                fcw_data = fcw_system.update(det["id"], bev_x, bev_y, w, h, dt)

                color = (0, 0, 255) if fcw_data["warning"] else (0, 255, 0)

                cv2.circle(
                    frame,
                    (int(bottom_center[0]), int(bottom_center[1])),
                    5,
                    (0, 0, 255),
                    -1,
                )

                if 0 <= bev_x < w and 0 <= bev_y < h:
                    cv2.circle(
                        bev_view,
                        (int(bev_x), int(bev_y)),
                        8,
                        (0, 0, 255),
                        -1,
                    )

                label = f"#{det['id']} {det["class_name"]} ({det['confidence']})"
                if fcw_data["distance_m"] is not None:
                    label += f"| {fcw_data['distance_m']}m"
                if fcw_data["ttc"] is not None:
                    label += f" | TTC: {fcw_data["ttc"]}s"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

            if fcw_data["warning"]:
                cv2.putText(
                    frame,
                    "ATTENTION COLLISION - STOP!",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )

            pip_w, pip_h = w // 4, h // 4
            bev_small = cv2.resize(bev_view, (pip_w, pip_h))
            frame[10 : 10 + pip_h, w - pip_w - 10 : w - 10] = bev_small

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
