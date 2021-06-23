from typing import Optional, Awaitable, Tuple

import numpy as np

from v4ray_frontend.camera import CameraLike
from v4ray_frontend.material import MaterialLike
from v4ray_frontend.shape import ShapeLike
from . import shape
from . import material
from . import texture

class Scene:
    def __new__(cls, background: Tuple[float, float, float]) -> Scene: ...
    def add(self, shape: ShapeLike, material: MaterialLike) -> None: ...


class AABB:
    def __new__(
            cls,
            min: Tuple[float, float, float],
            max: Tuple[float, float, float]
    ) -> AABB: ...
    @property
    def min(self) -> Tuple[float, float, float]: ...
    @property
    def max(self) -> Tuple[float, float, float]: ...


class Ray:
    # ... -> [f32; 3 x LANES]
    @property
    def origin(self) -> np.ndarray: ...
    # ... -> [f32; 3 x LANES]
    @property
    def direction(self) -> np.ndarray: ...
    # ... -> [f32; LANES]
    @property
    def time(self) -> np.ndarray: ...
    # ... -> [bool; LANES]
    @property
    def mask(self) -> np.ndarray: ...


class HitRecord:
    ...


class PerspectiveCameraParam(CameraLike):
    def __new__(
            cls,
            look_from: Tuple[float, float, float],
            look_at: Tuple[float, float, float],
            vfov: float,
            up: Optional[Tuple[float, float, float]] = None,
            aspect_ratio: Optional[float] = None,
            aperture: Optional[float] = None,
            focus_dist: Optional[float] = None,
            time0: Optional[float] = None,
            time1: Optional[float] = None,
    ) -> PerspectiveCameraParam: ...


class RendererParam:
    def __new__(
            cls,
            width: int,
            height: int,
            max_depth: Optional[int] = None,
    ) -> RendererParam: ...


class Renderer:
    def __new__(
            cls,
            param: RendererParam,
            camera: CameraLike,
            scene: Scene,
    ) -> Renderer: ...
    # ... -> [f32; height x width x 3]
    def render(self) -> Awaitable[np.ndarray]: ...


x: int = 1
