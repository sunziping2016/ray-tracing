#![allow(incomplete_features)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![feature(array_map)]

use async_channel::Sender;
use core::array;
use gtk::{ContainerExt, FrameExt, GtkWindowExt, ImageExt, WidgetExt};
use itertools::iproduct;
use nalgebra::{SimdBool, SimdRealField, SimdValue, Vector3};
use num_traits::{cast, clamp, Zero};
use rand::{thread_rng, Rng};
use ray_tracing::bvh::bvh::BVH;
use ray_tracing::camera::{Camera, CameraParam};
use ray_tracing::hittable::sphere::Sphere;
use ray_tracing::hittable::{HitRecord, Hittable};
use ray_tracing::image::ImageParam;
use ray_tracing::material::lambertian::Lambertian;
use ray_tracing::ray::Ray;
use ray_tracing::simd::MyFromSlice;
use ray_tracing::texture::solid_color::SolidColor;
use ray_tracing::{SimdBoolField, SimdF32Field};
use serde::{Deserialize, Serialize};
use simba::simd::f32x8;
use std::fmt::Debug;
use std::fs::File;
use std::future::Future;
use std::process;
use std::sync::{mpsc, Arc, RwLock};

#[derive(Debug, Clone)]
enum Event {
    SizeAllocated { width: i32, height: i32 },
    RedrawTimeout,
}

pub fn spawn<F>(future: F)
where
    F: Future<Output = ()> + 'static,
{
    glib::MainContext::default().spawn_local(future);
}

struct App {
    pub image: gtk::Image,
}

impl App {
    pub fn new(tx: Sender<Event>, width: i32, height: i32, aspect_ratio: f32) -> Self {
        let image = gtk::Image::new();

        let scrolled_window =
            gtk::ScrolledWindow::new::<gtk::Adjustment, gtk::Adjustment>(None, None);
        scrolled_window.add(&image);
        scrolled_window.set_size_request(width, height);
        scrolled_window.connect_size_allocate(move |_, event| {
            let width = event.width;
            let height = event.height;
            let tx = tx.clone();
            spawn(async move {
                let _ = tx
                    .send(Event::SizeAllocated {
                        width: width as i32,
                        height: height as i32,
                    })
                    .await;
            });
        });

        let container = gtk::AspectFrame::new(None, 0.5, 0.5, aspect_ratio, false);
        container.set_shadow_type(gtk::ShadowType::None);
        container.override_background_color(gtk::StateFlags::NORMAL, Some(&gdk::RGBA::black()));
        container.add(&scrolled_window);
        container.show_all();

        let window = gtk::Window::new(gtk::WindowType::Toplevel);
        window.set_title("Ray Tracing By Sun");
        window.add(&container);
        window.connect_delete_event(move |_, _| {
            gtk::main_quit();
            gtk::Inhibit(false)
        });
        window.show_all();
        gtk::Window::set_default_icon_name("icon-name-here");
        Self { image }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneParam {
    image: ImageParam,
    camera: CameraParam,
}

#[derive(Debug)]
pub struct Render<F>
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    scene: SceneParam,
    camera: Camera<F>,
    hittables: Vec<Sphere>,
    materials: Vec<Lambertian<SolidColor>>,
    bvh: BVH,
    screen: RwLock<(Vec<Vector3<F>>, usize)>,
    background: Vector3<f32>,
}

const NUM: usize = 200;

impl<F> Render<F>
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    pub fn new(scene: SceneParam) -> Self {
        let num_pixels = scene.image.height() * scene.image.width();
        let num_soa = (num_pixels + F::lanes() - 1) / F::lanes();
        let default_aspect_ratio = scene.image.aspect_ratio();
        let camera_param = scene.camera.clone();
        let mut rng = thread_rng();
        let (mut hittables, mut materials): (Vec<_>, Vec<_>) = (0..NUM)
            .map(|shape_index| {
                let x = shape_index as f32 / (NUM - 1) as f32;
                let color = Vector3::new(0.55, 0.9 - 0.7 * x, 0.2 + 0.7 * x);
                (
                    Sphere::new(
                        Vector3::new(
                            rng.gen_range(-300f32..300f32),
                            rng.gen_range(-300f32..300f32),
                            rng.gen_range(-300f32..300f32),
                        ),
                        18.0f32,
                    ),
                    Lambertian::new(SolidColor::new(color)),
                )
            })
            .unzip();
        hittables.push(Sphere::new(Vector3::new(0f32, -3318f32, 0f32), 3018f32));
        materials.push(Lambertian::new(SolidColor::new(Zero::zero())));
        let bvh = BVH::build(
            &hittables
                .iter()
                .map(|x| x.bounding_box(0.0f32, 0.0f32))
                .collect::<Vec<_>>(),
        );
        Self {
            scene,
            camera: Camera::new(camera_param, default_aspect_ratio),
            screen: RwLock::new((vec![Zero::zero(); num_soa], 0)),
            hittables,
            materials,
            bvh,
            background: Vector3::new(1.0f32, 1.0f32, 1.0f32),
        }
    }
    pub fn ray_color(&self, rays: Vec<Ray<F>>, depth: u64) -> Vec<Vector3<F>>
    where
        [(); F::LANES]: Sized,
    {
        if depth == 0 {
            return vec![Zero::zero(); rays.len()];
        }
        let mut hit_shapes: Vec<(Vec<usize>, Vec<Ray<F>>)> =
            vec![(Vec::new(), Vec::new()); self.hittables.len()];
        rays.iter().enumerate().for_each(|(index1, ray)| {
            array::IntoIter::new(
                self.bvh
                    .traverse(ray, F::splat(0f32), F::splat(f32::INFINITY)),
            )
            .enumerate()
            .filter(move |(index2, indices)| ray.mask().extract(*index2) && !indices.is_empty())
            .for_each(|(index2, indices)| {
                let index = index1 * F::LANES + index2;
                indices.iter().for_each(|shape_index| {
                    let (inserted_indices, inserted_rays) = &mut hit_shapes[*shape_index];
                    let inserted_index2 = inserted_indices.len() % F::LANES;
                    if inserted_index2 == 0 {
                        let mut inserted_ray = Ray::default();
                        inserted_ray.replace(0, ray.extract(index2));
                        inserted_rays.push(inserted_ray);
                    } else {
                        inserted_rays
                            .last_mut()
                            .unwrap()
                            .replace(inserted_index2, ray.extract(index2));
                    }
                    inserted_indices.push(index);
                });
            });
        });
        // (shape_index, hit_record)
        let mut hit_records =
            vec![(usize::MAX, HitRecord::<f32>::default()); rays.len() * F::LANES];
        let mut hit_record_indices = Vec::new();
        hit_shapes
            .into_iter()
            .enumerate()
            .for_each(|(shape_index, (ray_indices, rays))| {
                let shape = &self.hittables[shape_index];
                rays.into_iter().enumerate().for_each(|(index1, ray)| {
                    let hit_record = shape.hit(&ray, F::zero(), F::splat(f32::INFINITY));
                    if hit_record.mask.any() {
                        (0..F::LANES).for_each(|index2| {
                            if hit_record.mask.extract(index2) {
                                let ray_index = ray_indices[index1 * F::LANES + index2];
                                let record = &mut hit_records[ray_index];
                                if record.1.t > hit_record.t.extract(index2) {
                                    if !record.1.mask {
                                        hit_record_indices.push(ray_index);
                                    }
                                    record.0 = shape_index;
                                    record.1 = hit_record.extract(index2);
                                }
                            }
                        });
                    }
                });
            });
        println!("{}", hit_record_indices.len());
        let colors = vec![Vector3::splat(self.background); rays.len()];
        colors
    }
    pub fn run(&self)
    where
        F: MyFromSlice,
        [(); F::LANES]: Sized,
    {
        let mut rng = thread_rng();
        let rays = self
            .scene
            .image
            .sample::<F, _>(&mut rng)
            .into_iter()
            .map(|(st, mask)| self.camera.get_ray(st, mask, &mut rng))
            .collect::<Vec<_>>();
        let colors = self.ray_color(rays, 20);
        let mut lock = self.screen.write().unwrap();
        lock.0
            .iter_mut()
            .zip(colors.into_iter())
            .for_each(|(sum, v)| {
                *sum += v;
            });
        lock.1 += 1;
        println!("Iter {}", lock.1)
    }
    pub fn get(&self, last: usize) -> Option<(gdk_pixbuf::Pixbuf, usize)>
    where
        F: SimdRealField<Element = f32>,
    {
        let lock = self.screen.read().unwrap();
        let new_last = lock.1;
        if new_last <= last {
            return None;
        }
        let scale = F::splat(256.0 / lock.1 as f32);
        let colors = lock.0.iter().map(|x| x.scale(scale)).collect::<Vec<_>>();
        drop(lock);
        let height = self.scene.image.height();
        let width = self.scene.image.width();
        let mut bytes: Vec<u8> = vec![255; height * width * 3];
        let min: <F as SimdValue>::Element = cast(0.5).unwrap();
        let max: <F as SimdValue>::Element = cast(255.5).unwrap();
        iproduct!(0..height, 0..width).for_each(|(y, x)| {
            let index = y * width + x;
            let (index1, index2) = (index / f32x8::lanes(), index % f32x8::lanes());
            let base = index * 3;
            let color = colors[index1].map(|x| cast(clamp(x.extract(index2), min, max)).unwrap());
            (0..2).for_each(|index| bytes[base + index] = color[index]);
        });
        Some((
            gdk_pixbuf::Pixbuf::from_bytes(
                &glib::Bytes::from(bytes.as_slice()),
                gdk_pixbuf::Colorspace::Rgb,
                false,
                8,
                width as i32,
                height as i32,
                width as i32 * 3,
            ),
            new_last,
        ))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rayon::ThreadPoolBuilder::new()
        .start_handler(|_| {
            glib::MainContext::new().push_thread_default();
        })
        .build_global()?;

    glib::set_program_name("Ray Tracing By Sun".into());
    glib::set_application_name("Ray Tracing By Sun");
    if gtk::init().is_err() {
        eprintln!("failed to initialize GTK Application");
        process::exit(1);
    }

    let state: Arc<Render<f32x8>> = Arc::new(Render::new(serde_json::from_reader(File::open(
        "data/scene.json",
    )?)?));
    let (msg_tx, msg_rx) = async_channel::bounded(16);
    let (render_tx, render_rx) = mpsc::channel();

    let aspect_ratio = state.scene.image.width() as f32 / state.scene.image.height() as f32;
    let app = App::new(msg_tx.clone(), 800, 600, aspect_ratio);
    // Processes all application events received from signals
    {
        let state = state.clone();
        glib::MainContext::default().spawn_local(async move {
            let mut pixbuf: Option<gdk_pixbuf::Pixbuf> = None;
            let mut width = i32::MAX;
            let mut height = i32::MAX;
            let mut last = 0;
            // let mut num = 0usize;
            while let Ok(event) = msg_rx.recv().await {
                match event {
                    Event::SizeAllocated {
                        width: new_width,
                        height: new_height,
                    } => {
                        if let Some(pixbuf) = pixbuf.as_ref() {
                            if width != new_width || height != new_height {
                                width = new_width;
                                height = new_height;
                                app.image.set_from_pixbuf(
                                    pixbuf
                                        .scale_simple(
                                            width,
                                            height,
                                            gdk_pixbuf::InterpType::Nearest,
                                        )
                                        .as_ref(),
                                );
                                app.image.set_size_request(0, 0);
                            }
                        }
                    }
                    Event::RedrawTimeout => {
                        if let Some((new_pixbuf, new_last)) = state.get(last) {
                            app.image.set_from_pixbuf(
                                new_pixbuf
                                    .scale_simple(width, height, gdk_pixbuf::InterpType::Nearest)
                                    .as_ref(),
                            );
                            app.image.set_size_request(0, 0);
                            println!("redraw");
                            pixbuf = Some(new_pixbuf);
                            last = new_last;
                        }
                    }
                }
            }
        });
    }

    // Trigger rendering
    let spawn_render = move || {
        let state = state.clone();
        let render_tx = render_tx.clone();
        rayon::spawn(move || {
            state.run();
            let _ = render_tx.send(());
        });
    };
    rayon::spawn(move || {
        for _ in 0..num_cpus::get() {
            spawn_render();
        }
        spawn_render();
        while render_rx.recv().is_ok() {
            spawn_render();
        }
    });

    // Trigger drawing
    {
        let _ = glib::source::timeout_add(1000 / 30, move || {
            let tx = msg_tx.clone();
            spawn(async move {
                let _ = tx.send(Event::RedrawTimeout).await;
            });
            glib::Continue(true)
        });
    }
    // Rendering
    gtk::main();
    Ok(())
}
