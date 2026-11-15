import cv2
import numpy as np


class BirdEyeTransformer:

    def __init__(self, img_with: int = 1280, img_height: int = 720):
        self.w = img_with
        self.h = img_height

        src = np.float32(
            [
                [self.w * 0.43, self.h * 0.62],
                [self.w * 0.57, self.h * 0.62],
                [self.w * 0.85, self.h * 0.95],
                [self.w * 0.15, self.h * 0.95],
            ]
        )

        dst = np.float32(
            [
                [self.w * 0.25, 0],
                [self.w * 0.75, 0],
                [self.w * 0.75, self.h],
                [self.w * 0.25, self.h],
            ]
        )

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def warp_to_bev(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if w != self.w or h != self.h:
            self.__init__(w, h)
        return cv2.warpPerspective(frame, self.M, (w, h), flags=cv2.INTER_LINEAR)

    def transform_point(self, point: tuple[float, float]):
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        tranformed = cv2.perspectiveTransform(pt, self.M)
        return float(tranformed[0][0][0]), float(tranformed[0][0][1])
