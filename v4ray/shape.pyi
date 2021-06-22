from typing import Tuple

import numpy as np

from v4ray import Ray, HitRecord, AABB


class Sphere:
    def __new__(
            cls,
            center: Tuple[float, float, float],
            radius: float
    ) -> Sphere: ...
    @property
    def center(self) -> Tuple[float, float, float]: ...
    @property
    def radius(self) -> float: ...
    def bounding_box(self) -> AABB: ...
    def hit(self, ray: Ray, t_min: np.ndarray,
            t_max: np.ndarray) -> HitRecord: ...