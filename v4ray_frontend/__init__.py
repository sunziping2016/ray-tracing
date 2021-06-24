from typing import Type, List

from .camera import CameraType, PerspectiveCamera
from .material import MaterialType, Lambertian, Metal, Dielectric
from .shape import Sphere, ShapeType
from .texture import TextureType, SolidColor, Checker

shapes: List[Type[ShapeType]] = [Sphere]
textures: List[Type[TextureType]] = [SolidColor, Checker]
materials: List[Type[MaterialType]] = [Lambertian, Metal, Dielectric]
cameras: List[Type[CameraType]] = [PerspectiveCamera]
