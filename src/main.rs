use async_std::task;
use image::{ImageBuffer, Rgb};
use nalgebra::{convert, Point3, Rotation3, Translation3, UnitVector3, Vector3};
use num_traits::Zero;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use simba::simd::f32x8;
use std::error::Error;
use std::fs::File;
use std::sync::{mpsc, Arc};
use std::time::{Duration, SystemTime};
use v4ray::camera::CameraParam;
use v4ray::hittable::aa_rect::{XYRect, YZRect, ZXRect};
use v4ray::hittable::sphere::Sphere;
use v4ray::hittable::transform::TransformHittable;
use v4ray::hittables::cuboid::Cuboid;
use v4ray::material::dielectric::Dielectric;
use v4ray::material::diffuse_light::DiffuseLight;
use v4ray::material::lambertian::Lambertian;
use v4ray::renderer::{RenderResult, Renderer, RendererParam};
use v4ray::scene::Scene;
use v4ray::texture::solid_color::SolidColor;
use v4ray::BoxedHittable;

type F = f32x8;
type R = ThreadRng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneParam {
    renderer: RendererParam,
    camera: CameraParam,
}

#[async_std::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut scene = Scene::new(Vector3::new(0f32, 0f32, 0f32), Zero::zero());
    let red = Arc::new(Lambertian::new(SolidColor::new(Vector3::new(
        0.65, 0.05, 0.05,
    ))));
    let white = Arc::new(Lambertian::new(SolidColor::new(Vector3::new(
        0.73, 0.73, 0.73,
    ))));
    let green = Arc::new(Lambertian::new(SolidColor::new(Vector3::new(
        0.12, 0.45, 0.15,
    ))));
    let light = Arc::new(DiffuseLight::new(SolidColor::new(Vector3::new(
        15., 15., 15.,
    ))));
    scene.add(
        Arc::new(YZRect::new(0., 555., 0., 555., 555., false)),
        green,
    );
    scene.add(Arc::new(YZRect::new(0., 555., 0., 555., 0., true)), red);
    let light_hittable = Arc::new(ZXRect::new(213., 343., 227., 332., 554., false));
    scene.add_important(light_hittable.clone(), light_hittable, light);
    scene.add(
        Arc::new(ZXRect::new(0., 555., 0., 555., 0., true)),
        white.clone(),
    );
    scene.add(
        Arc::new(ZXRect::new(0., 555., 0., 555., 555., false)),
        white.clone(),
    );
    scene.add(
        Arc::new(XYRect::new(0., 555., 0., 555., 555., false)),
        white.clone(),
    );
    // Boxes
    scene.add_all(
        Cuboid::new(Vector3::new(0., 0., 0.), Vector3::new(165., 330., 165.))
            .into_iter()
            .map(|x| {
                Arc::new(TransformHittable::new(
                    convert(
                        Translation3::new(265., 0., 295.)
                            * Rotation3::from_axis_angle(
                                &UnitVector3::new_unchecked(Vector3::y()),
                                15f32.to_radians(),
                            ),
                    ),
                    x,
                )) as BoxedHittable<F, R>
            }),
        white.clone(),
    );
    scene.add(
        Arc::new(Sphere::new(Point3::new(190., 90., 190.), 90.0)),
        Arc::new(Dielectric::new(1.5)),
    );
    let param: SceneParam = serde_json::from_reader(File::open("data/scene.json")?)?;
    let width = param.renderer.width;
    let height = param.renderer.height;
    let renderer = Arc::new(Renderer::new(param.renderer, param.camera, scene));
    let render_result = Arc::new(RenderResult::new(width, height));
    let start = Arc::new(SystemTime::now());

    let (render_tx, render_rx) = mpsc::channel();
    let render_result1 = render_result.clone();
    let spawn_render = move || {
        let renderer = renderer.clone();
        let render_result = render_result1.clone();
        let render_tx = render_tx.clone();
        let start = start.clone();
        rayon::spawn(move || {
            let mut rng = thread_rng();
            println!(
                "Iter {} +{}s",
                render_result.add(renderer.render(&mut rng)),
                start.elapsed().unwrap().as_secs()
            );
            let _ = render_tx.send(());
        });
    };
    rayon::spawn(move || {
        for _ in 0..num_cpus::get() {
            spawn_render();
        }
        while render_rx.recv().is_ok() {
            spawn_render();
        }
    });

    task::spawn(async move {
        let mut last = 0;
        loop {
            task::sleep(Duration::from_secs(1)).await;
            if let Some((bytes, new_last)) = render_result.get_raw(last) {
                if let Ok(()) = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, bytes)
                    .unwrap()
                    .save("data/output.bmp")
                {
                    println!("Iter {} saved", new_last);
                }
                last = new_last;
            }
        }
    });
    let mut line = String::new();
    println!("Press enter to exit.");
    let _ = async_std::io::stdin().read_line(&mut line).await;
    Ok(())
}
