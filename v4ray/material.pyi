from v4ray_frontend.texture import TextureLike


class Lambertian:
    def __new__(cls, texture: TextureLike) -> Lambertian: ...
