from v4ray import TextureLike


class Lambertian:
    def __new__(cls, texture: TextureLike) -> Lambertian: ...
