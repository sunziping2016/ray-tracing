import random
import numpy as np
import cv2
import asyncio
from v4ray import *

async def main():
    width = 1200
    height = 800

    scene = Scene(bg_color=(1, 1, 1))
    scene.add(
        shape.Sphere((0, -1000, 0), 1000),
        material.Lambertian(texture.SolidColor((0.5, 0.5, 0.5)))
    )
    for a in range(-11, 11):
        for b in range(-11, 11):
            s = shape.Sphere((
                a + 0.9 * random.random(),
                0.2,
                b + 0.9 * random.random(),
            ), 0.2)
            m = material.Lambertian(texture.SolidColor((
                random.random(),
                random.random(),
                random.random()
            )))
            scene.add(s, m)
    camera = CameraParam(
        look_from=(13, 2, -3),
        look_at=(0, 0, 0),
        vfov=20,
        aperture=0.1,
        focus_dist=10
    )
    renderer = Renderer(
        RendererParam(width=width, height=height, max_depth=50),
        camera, scene,
    )
    result = np.empty((height, width, 3))
    count = 0
    while True:
        data = await renderer.render()
        result += data
        count += 1
        print(f'Iter: {count}')
        print(data.shape)
        cv2.imshow('Test', cv2.cvtColor((result / count).astype(np.float32), cv2.COLOR_RGB2BGR))
        cv2.waitKey()

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
