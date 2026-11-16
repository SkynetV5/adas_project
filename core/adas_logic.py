class CollisionWarningSystem:
    def __init__(self, pixels_per_meter: float = 62.0, ttc_threshold: float = 2.5):
        self.pixels_per_meter = pixels_per_meter
        self.ttc_threshold = ttc_threshold

        self.prev_distances = {}

    def update(
        self,
        track_id: int,
        bev_x: float,
        bev_y: float,
        img_width: int,
        img_height: int,
        dt: float,
    ):

        if not (0.0 < bev_y < img_height):
            self.prev_distances.pop(track_id, None)
            return {
                "distance_m": None,
                "ttc": None,
                "warning": False,
                "in_ego_lane": False,
            }
        raw_distance_m = max((img_height - bev_y) / self.pixels_per_meter, 0.0)

        ttc = float("inf")
        warning = False

        in_ego_lane = (img_width * 0.35) <= bev_x <= (img_width * 0.65)

        if track_id in self.prev_distances and dt > 0:
            prev_dist = self.prev_distances[track_id]
            distance_m = 0.7 * prev_dist + 0.3 * raw_distance_m
            closing_speed = (prev_dist - distance_m) / dt

            if in_ego_lane and closing_speed > 0.5:
                ttc = distance_m / closing_speed
                if ttc < self.ttc_threshold:
                    warning = True
        else:
            distance_m = raw_distance_m

        self.prev_distances[track_id] = distance_m

        return {
            "distance_m": round(distance_m, 1),
            "ttc": round(ttc, 1) if ttc != float("inf") else None,
            "warning": warning,
            "in_ego_line": in_ego_lane,
        }
