import cv2
from ultralytics import YOLO


class ObjectTracker:

    def __init__(self, model_path: str = "yolo11n.pt"):

        self.model = YOLO(model_path)

        # 0: person, 1: bike, 2: car, 3: motorbike, 5: bus, 7: truck, 9: traffic lights
        self.target_classes = [0, 1, 2, 3, 4, 5, 7, 9]

    def process_frame(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=self.target_classes,
            verbose=False,
        )

        detections = []
        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:
            coords = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()

            for box, track_id, cls_id, conf in zip(coords, track_ids, classes, confs):
                detections.append(
                    {
                        "id": track_id,
                        "class_id": cls_id,
                        "class_name": self.model.names[cls_id],
                        "bbox": [int(v) for v in box],
                        "confidence": round(conf, 2),
                    }
                )
        return detections
