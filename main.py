import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow(video_path="data/dashcam.mp4")
    window.show()
    sys.exit(app.exec())
