#![allow(incomplete_features)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![feature(array_map)]

use arrayvec::ArrayVec;
use async_channel::Sender;
use core::array;
use gtk::{ContainerExt, FrameExt, GtkWindowExt, ImageExt, WidgetExt};
use itertools::iproduct;
use nalgebra::{SimdRealField, SimdValue, Vector3};
use num_traits::{cast, clamp, Zero};
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng, SeedableRng};
use rand_pcg::Pcg64;
use ray_tracing::bvh::bvh::BVH;
use ray_tracing::camera::{Camera, CameraParam};
use ray_tracing::hittable::sphere::Sphere;
use ray_tracing::hittable::{HitRecord, Hittable};
use ray_tracing::image::ImageParam;
use ray_tracing::material::dielectric::Dielectric;
use ray_tracing::material::lambertian::Lambertian;
use ray_tracing::material::metal::Metal;
use ray_tracing::material::{Material, ScatterRecord};
use ray_tracing::ray::Ray;
use ray_tracing::texture::solid_color::SolidColor;
use ray_tracing::{extract, EPSILON};
use ray_tracing::{SimdBoolField, SimdF32Field};
use serde::{Deserialize, Serialize};
use std::array::IntoIter;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::fs::File;
use std::future::Future;
use std::iter::FromIterator;
use std::process;
use std::sync::{mpsc, Arc, RwLock};
use std::time::SystemTime;

type Float = simba::simd::f32x8;

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
    #[serde(skip_serializing_if = "Option::is_none", default)]
    max_depth: Option<u64>,
}

pub struct Render<F>
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    scene: SceneParam,
    camera: Camera<F>,
    hittables: Vec<Sphere>,
    materials: Vec<Box<dyn Material<F, ThreadRng> + Send + Sync>>,
    bvh: BVH,
    screen: RwLock<(Vec<Vector3<F>>, usize)>,
    background: Vector3<f32>,
    start: SystemTime,
}

const NUM: usize = 200;

impl<F> Render<F>
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    pub fn new(scene: SceneParam) -> Self {
        let metallic_albedo = HashMap::<_, _>::from_iter(IntoIter::new([
            // ("Iron", Vector3::new(198u8, 198u8, 200u8)),
            // ("Brass", Vector3::new(214u8, 185u8, 123u8)),
            ("Copper", Vector3::new(250u8, 208u8, 192u8)),
            // ("Gold", Vector3::new(255u8, 226u8, 155u8)),
            // ("Aluminium", Vector3::new(245u8, 246u8, 246u8)),
            // ("Chrome", Vector3::new(196u8, 197u8, 197u8)),
            // ("Silver", Vector3::new(252u8, 250u8, 245u8)),
            // ("Cobalt", Vector3::new(211u8, 210u8, 207u8)),
            // ("Titanium", Vector3::new(195u8, 186u8, 177u8)),
            // ("Platinum", Vector3::new(213u8, 208u8, 200u8)),
            // ("Nickel", Vector3::new(211u8, 203u8, 190u8)),
            // ("Zinc", Vector3::new(213u8, 234u8, 237u8)),
            // ("Mercury", Vector3::new(229u8, 228u8, 228u8)),
            // ("Palladium", Vector3::new(222u8, 217u8, 211u8)),
        ]));
        let metallic_albedo = metallic_albedo
            .into_iter()
            .map(|(_name, albedo)| albedo.map(|x| x as f32 / 255f32))
            .collect::<Vec<_>>();

        let num_pixels = scene.image.height() * scene.image.width();
        let num_soa = (num_pixels + F::lanes() - 1) / F::lanes();
        let default_aspect_ratio = scene.image.aspect_ratio();
        let camera_param = scene.camera.clone();
        let mut rng = Pcg64::seed_from_u64(2);
        let (mut hittables, mut materials): (Vec<_>, Vec<_>) = (0..NUM)
            .map(|shape_index| {
                let x = shape_index as f32 / (NUM - 1) as f32;
                let material = if rng.gen_ratio(30, 100) {
                    let color = Vector3::new(0.0, 0.9 - 0.5 * x, 0.4 + 0.5 * x);
                    Box::new(Lambertian::new(SolidColor::new(color)))
                        as Box<dyn Material<F, ThreadRng> + Send + Sync>
                } else if rng.gen_ratio(35, 70) {
                    let color = metallic_albedo[rng.gen_range(0..metallic_albedo.len())];
                    Box::new(Metal::new(color, rng.gen_range(0.1f32..0.7f32)))
                        as Box<dyn Material<F, ThreadRng> + Send + Sync>
                } else {
                    Box::new(Dielectric::new(1.5)) as Box<dyn Material<F, ThreadRng> + Send + Sync>
                };
                (
                    Sphere::new(
                        Vector3::new(
                            rng.gen_range(-300f32..300f32),
                            rng.gen_range(-300f32..300f32),
                            rng.gen_range(-300f32..300f32),
                        ),
                        30.0f32,
                    ),
                    material,
                )
            })
            .unzip();
        hittables.push(Sphere::new(Vector3::new(0f32, -20336f32, 0f32), 20000f32));
        materials.push(Box::new(Lambertian::new(SolidColor::new(Vector3::new(
            0.8, 0.8, 0.8,
        )))));
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
            start: SystemTime::now(),
        }
    }
    pub fn ray_color(&self, rays: Vec<Ray<F>>, depth: u64, rng: &mut ThreadRng) -> Vec<Vector3<F>>
    where
        [(); F::LANES]: Sized,
        F: From<[f32; F::LANES]>,
        F::SimdBool: From<[bool; F::LANES]>,
    {
        if depth == 0 {
            return vec![Zero::zero(); rays.len()];
        }
        let mut hit_shapes: Vec<(Vec<usize>, Vec<Ray<f32>>)> =
            vec![(Vec::new(), Vec::new()); self.hittables.len()];
        rays.iter().enumerate().for_each(|(index1, ray)| {
            array::IntoIter::new(
                self.bvh
                    .traverse(ray, F::splat(EPSILON), F::splat(f32::INFINITY)),
            )
            .enumerate()
            .filter(move |(index2, indices)| unsafe { ray.mask().extract_unchecked(*index2) } && !indices.is_empty())
            .for_each(|(index2, indices)| {
                let ray_index = index1 * F::LANES + index2;
                indices.iter().for_each(|shape_index| {
                    let (inserted_indices, inserted_rays) = &mut hit_shapes[*shape_index];
                    inserted_rays.push(unsafe { ray.extract_unchecked(index2) });
                    inserted_indices.push(ray_index);
                });
            });
        });
        let mut hit_records =
            vec![(usize::MAX, HitRecord::<f32>::default()); rays.len() * F::LANES];
        let mut hit_ray_indices = Vec::new();
        hit_shapes
            .into_iter()
            .enumerate()
            .for_each(|(shape_index, (ray_indices, mut rays))| {
                let shape = &self.hittables[shape_index];
                let rays_padding = rays.len() % F::LANES;
                if rays_padding != 0 {
                    rays.extend((0..(F::LANES - rays_padding)).map(|_| Ray::default()));
                }
                (0..rays.len())
                    .step_by(F::LANES)
                    .into_iter()
                    .for_each(|index1| {
                        let ray = Ray::<F>::from(&rays[index1..(index1 + F::LANES)]);
                        let hit_record =
                            shape.hit(&ray, F::splat(EPSILON), F::splat(f32::INFINITY));
                        (0..F::LANES)
                            .take_while(|index2| unsafe { ray.mask().extract_unchecked(*index2) })
                            .filter(|index2| unsafe { hit_record.mask.extract_unchecked(*index2) })
                            .for_each(|index2| {
                                let ray_index = ray_indices[index1 + index2];
                                let record = &mut hit_records[ray_index];
                                if record.1.t > unsafe { hit_record.t.extract_unchecked(index2) } {
                                    if !record.1.mask {
                                        hit_ray_indices.push(ray_index);
                                    }
                                    record.0 = shape_index;
                                    record.1 = unsafe { hit_record.extract_unchecked(index2) };
                                }
                            });
                    });
            });
        let mut colors = vec![Vector3::splat(self.background); rays.len()];
        if hit_ray_indices.is_empty() {
            return colors;
        }
        // shape_index => (ray_indices, rays, hit_records)
        let mut hit_shapes: BTreeMap<usize, (_, Vec<Ray<f32>>, _)> = BTreeMap::new();
        hit_ray_indices.into_iter().for_each(|ray_index| {
            let ray = unsafe { rays[ray_index / F::LANES].extract_unchecked(ray_index % F::LANES) };
            let (shape_index, hit_record) = hit_records[ray_index].clone();
            let (ray_indices, rays, hit_records) = hit_shapes
                .entry(shape_index)
                .or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
            rays.push(ray);
            hit_records.push(hit_record);
            ray_indices.push(ray_index);
        });
        // (rays, (coef, emitted, indices))
        let mut scattered_rays = Vec::new();
        let mut scattered_coef = Vec::new();
        let mut scattered_indices = Vec::new();
        hit_shapes.into_iter().for_each(
            |(shape_index, (ray_indices, mut rays, mut hit_records))| {
                let material = &self.materials[shape_index];
                assert_eq!(rays.len(), hit_records.len());
                let rays_padding = rays.len() % F::LANES;
                if rays_padding != 0 {
                    rays.extend((0..(F::LANES - rays_padding)).map(|_| Ray::default()));
                    hit_records
                        .extend((0..(F::LANES - rays_padding)).map(|_| HitRecord::default()));
                }
                (0..rays.len()).step_by(F::LANES).for_each(|index1| {
                    let ray = Ray::<F>::from(&rays[index1..(index1 + F::LANES)]);
                    let hit_record =
                        HitRecord::<F>::from(&hit_records[index1..(index1 + F::LANES)]);
                    let emitted = material.emitted(&hit_record.uv, &hit_record.p);
                    let mask = ray.mask();
                    (0..F::LANES)
                        .take_while(|index2| unsafe { mask.extract_unchecked(*index2) })
                        .for_each(|index2| {
                            let ray_index = ray_indices[index1 + index2];
                            unsafe {
                                colors[ray_index / F::LANES].replace_unchecked(
                                    ray_index % F::LANES,
                                    emitted.extract_unchecked(index2),
                                )
                            }
                        });
                    let (attenuation, pdf) = {
                        match material.scatter(&ray, &hit_record, rng) {
                            ScatterRecord::Scatter { attenuation, pdf } => (attenuation, pdf),
                            ScatterRecord::Specular {
                                attenuation,
                                specular_ray,
                            } => {
                                (0..F::lanes())
                                    .take_while(|index2| unsafe { mask.extract_unchecked(*index2) })
                                    .for_each(|index2| {
                                        let ray_index = ray_indices[index1 + index2];
                                        scattered_rays.push(unsafe {
                                            specular_ray.extract_unchecked(index2)
                                        });
                                        scattered_coef
                                            .push(unsafe { attenuation.extract_unchecked(index2) });
                                        scattered_indices.push(ray_index);
                                    });
                                return;
                            }
                            _ => return,
                        }
                    };
                    let scattered =
                        Ray::new(hit_record.p, pdf.generate(rng), *ray.time(), ray.mask());
                    let coef = attenuation;
                    (0..F::lanes())
                        .take_while(|index2| unsafe { mask.extract_unchecked(*index2) })
                        .for_each(|index2| {
                            let ray_index = ray_indices[index1 + index2];
                            scattered_rays.push(unsafe { scattered.extract_unchecked(index2) });
                            scattered_coef.push(unsafe { coef.extract_unchecked(index2) });
                            scattered_indices.push(ray_index);
                        });
                });
            },
        );
        if scattered_indices.is_empty() {
            return colors;
        }
        let rays_padding = scattered_rays.len() % F::LANES;
        if rays_padding != 0 {
            scattered_rays.extend((0..(F::LANES - rays_padding)).map(|_| Ray::default()));
            scattered_coef
                .extend((0..(F::LANES - rays_padding)).map(|_| Vector3::from_element(f32::NAN)));
        }
        let scattered_rays = (0..scattered_rays.len())
            .step_by(F::LANES)
            .map(|index| Ray::<F>::from(&scattered_rays[index..(index + F::LANES)]))
            .collect::<Vec<_>>();
        let scattered_coef = (0..scattered_coef.len())
            .step_by(F::LANES)
            .map(|index| {
                let coef = &scattered_coef[index..(index + F::LANES)];
                let r = extract!(coef, |x| x[0]);
                let g = extract!(coef, |x| x[1]);
                let b = extract!(coef, |x| x[2]);
                Vector3::new(F::from(r), F::from(g), F::from(b))
            })
            .collect::<Vec<_>>();
        let new_colors = self.ray_color(scattered_rays, depth - 1, rng);
        new_colors
            .into_iter()
            .zip(scattered_coef.into_iter())
            .enumerate()
            .for_each(|(index1, (new_color, coef))| {
                let extra_color = new_color.component_mul(&coef);
                (0..F::LANES)
                    .map(|index2| (index2, index1 * F::LANES + index2))
                    .take_while(|(_, index)| *index < scattered_indices.len())
                    .for_each(|(index2, index)| {
                        let scattered_index = scattered_indices[index];
                        let ray_index1 = scattered_index / F::LANES;
                        let ray_index2 = scattered_index % F::LANES;
                        unsafe {
                            let color = colors[ray_index1].extract_unchecked(ray_index2)
                                + extra_color.extract_unchecked(index2);
                            colors[ray_index1].replace_unchecked(ray_index2, color);
                        }
                    })
            });
        colors
    }
    pub fn run(&self)
    where
        F: From<[f32; F::LANES]>,
        F::SimdBool: From<[bool; F::LANES]>,
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
        let colors = self.ray_color(rays, self.scene.max_depth.unwrap_or(20), &mut rng);
        let mut lock = self.screen.write().unwrap();
        lock.0
            .iter_mut()
            .zip(colors.into_iter())
            .for_each(|(sum, v)| {
                *sum += v;
            });
        lock.1 += 1;
        println!(
            "Iter {} +{}s",
            lock.1,
            self.start.elapsed().unwrap().as_secs()
        )
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
        let scale1 = F::splat(1.0 / lock.1 as f32);
        let scale2 = F::splat(256f32);
        let colors = lock.0.clone();
        drop(lock);
        let colors = colors
            .into_iter()
            .map(|x| x.scale(scale1).map(|x| x.simd_sqrt()).scale(scale2))
            .collect::<Vec<_>>();
        let height = self.scene.image.height();
        let width = self.scene.image.width();
        let mut bytes: Vec<u8> = vec![255; height * width * 3];
        let min: <F as SimdValue>::Element = cast(0.5).unwrap();
        let max: <F as SimdValue>::Element = cast(255.5).unwrap();
        iproduct!(0..height, 0..width).for_each(|(y, x)| {
            let index = y * width + x;
            let (index1, index2) = (index / F::lanes(), index % F::lanes());
            let base = index * 3;
            let color = colors[index1]
                .map(|x| cast(clamp(unsafe { x.extract_unchecked(index2) }, min, max)).unwrap());
            (0..3).for_each(|index| bytes[base + index] = color[index]);
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

    let state: Arc<Render<Float>> = Arc::new(Render::new(serde_json::from_reader(File::open(
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
                                            gdk_pixbuf::InterpType::Bilinear,
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
                                    .scale_simple(width, height, gdk_pixbuf::InterpType::Bilinear)
                                    .as_ref(),
                            );
                            app.image.set_size_request(0, 0);
                            println!("redraw");
                            new_pixbuf.savev("data/output.jpeg", "jpeg", &[]).unwrap();
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
        while render_rx.recv().is_ok() {
            spawn_render();
        }
    });

    // Trigger drawing
    {
        let _ = glib::source::timeout_add(1000, move || {
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
