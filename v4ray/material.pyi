from typing import Tuple

from v4ray_frontend.texture import TextureLike


class Lambertian:
    def __new__(cls, texture: TextureLike) -> Lambertian: ...


class Dielectric:
    def __new__(cls, ir: float) -> Dielectric: ...

class Metal:
    def __new__(cls, albedo: Tuple[float, float, float],
                fuzz: float) -> Metal: ...
