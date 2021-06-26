#![allow(incomplete_features)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![feature(array_map)]

use async_channel::Sender;
use gtk::{ContainerExt, FrameExt, GtkWindowExt, ImageExt, WidgetExt};
use image::io::Reader as ImageReader;
use nalgebra::{convert, Point3, Rotation3, SimdRealField, UnitVector3, Vector3};
use num_traits::Zero;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng, SeedableRng};
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fs::File;
use std::future::Future;
use std::io::{BufRead, BufReader};
use std::process;
use std::sync::{mpsc, Arc};
use std::time::SystemTime;
use v4ray::camera::CameraParam;
use v4ray::hittable::aa_rect::{XYRect, YZRect, ZXRect};
use v4ray::hittable::sphere::Sphere;
use v4ray::hittable::transform::TransformHittable;
use v4ray::hittable::triangle::Triangle;
use v4ray::hittables::cuboid::Cuboid;
use v4ray::material::dielectric::Dielectric;
use v4ray::material::diffuse_light::DiffuseLight;
use v4ray::material::lambertian::Lambertian;
use v4ray::material::metal::Metal;
use v4ray::renderer::{RenderResult, Renderer, RendererParam};
use v4ray::scene::Scene;
use v4ray::texture::checker::Checker;
use v4ray::texture::image::ImageTexture;
use v4ray::texture::noise::{Noise, Perlin};
use v4ray::texture::solid_color::SolidColor;
use v4ray::{BoxedHittable, SimdBoolField, SimdF32Field};

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
    renderer: RendererParam,
    camera: CameraParam,
}

pub struct Example<F, R: Rng>
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    renderer: Renderer<F, R>,
    result: RenderResult<F>,
    start: SystemTime,
}

impl<F, R: Rng> Example<F, R>
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    pub fn create_world<G: Rng>(rng: &mut G) -> Scene<F, R> {
        let mut scene = Scene::new(Vector3::new(1f32, 1f32, 1f32), Zero::zero());
        scene.add(
            Arc::new(Sphere::new(Point3::new(0f32, -1000f32, 0f32), 1000f32)),
            Arc::new(Lambertian::new(Checker::new(
                SolidColor::new(Vector3::new(0.2f32, 0.3f32, 0.1f32)),
                SolidColor::new(Vector3::new(0.9f32, 0.9f32, 0.9f32)),
                10f32,
            ))),
        );
        for a in -11..11 {
            for b in -11..11 {
                let center = Point3::new(
                    a as f32 + 0.9f32 * rng.gen_range(0f32..=1f32),
                    0.2f32,
                    b as f32 + 0.9f32 * rng.gen_range(0f32..=1f32),
                );
                if (center - Point3::new(4.0f32, 0.2f32, 0f32)).norm() > 0.9 {
                    let choose_mat = rng.gen_range(0f32..=1f32);
                    scene.add(
                        Arc::new(Sphere::new(center, 0.2f32)),
                        if choose_mat < 0.8 {
                            let albedo = Vector3::new(
                                rng.gen_range(0f32..=1f32),
                                rng.gen_range(0f32..=1f32),
                                rng.gen_range(0f32..=1f32),
                            );
                            let albedo = albedo.component_mul(&albedo);
                            Arc::new(Lambertian::new(SolidColor::new(albedo)))
                        } else if choose_mat < 0.95 {
                            let albedo = Vector3::new(
                                rng.gen_range(0.5f32..=1f32),
                                rng.gen_range(0.5f32..=1f32),
                                rng.gen_range(0.5f32..=1f32),
                            );
                            let fuzz = rng.gen_range(0f32..=0.5f32);
                            Arc::new(Metal::new(albedo, fuzz))
                        } else {
                            Arc::new(Dielectric::new(1.5f32))
                        },
                    );
                }
            }
        }
        scene.add(
            Arc::new(Sphere::new(Point3::new(0f32, 1f32, 0f32), 1f32)),
            Arc::new(Dielectric::new(1.5f32)),
        );
        scene.add(
            Arc::new(Sphere::new(Point3::new(-4f32, 1f32, 0f32), 1f32)),
            Arc::new(Lambertian::new(SolidColor::new(Vector3::new(
                0.4f32, 0.2f32, 0.1f32,
            )))),
        );
        scene.add(
            Arc::new(Sphere::new(Point3::new(4f32, 1f32, 0f32), 1f32)),
            Arc::new(Metal::new(Vector3::new(0.7f32, 0.6f32, 0.5f32), 0.0)),
        );
        scene
    }
    pub fn create_world2<G>(_rng: &mut G) -> Scene<F, R> {
        // let material = Arc::new(Lambertian::new(SolidColor::new(Vector3::new(
        //     0.4f32, 0.2f32, 0.1f32,
        // ))));
        let material = Arc::new(Metal::new(Vector3::new(0.7f32, 0.6f32, 0.5f32), 0.3));
        let mut vertices = Vec::new();
        let mut vertex_norms = Vec::new();
        let mut faces = Vec::new();
        let bunny_file = File::open("data/bunny.obj").unwrap();
        for line in BufReader::new(bunny_file).lines().map(|l| l.unwrap()) {
            if line.starts_with('v') {
                let vertex = line
                    .split(' ')
                    .skip(1)
                    .map(|x| x.parse::<f32>().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(vertex.len(), 3);
                vertices.push(Point3::new(vertex[0], vertex[1], vertex[2]));
                vertex_norms.push(Vector3::from_element(0.0f32));
            } else if line.starts_with('f') {
                let indices = line
                    .split(' ')
                    .skip(1)
                    .map(|x| x.parse::<usize>().unwrap() - 1)
                    .collect::<Vec<_>>();
                assert_eq!(indices.len(), 3);
                faces.push([indices[0], indices[1], indices[2]]);
                let vertex1 = vertices[indices[0]];
                let vertex2 = vertices[indices[1]];
                let vertex3 = vertices[indices[2]];
                let norm = (vertex2 - vertex1).cross(&(vertex3 - vertex2)).normalize();
                vertex_norms[indices[0]] += norm;
                vertex_norms[indices[1]] += norm;
                vertex_norms[indices[2]] += norm;
            }
        }
        let mut scene = Scene::new(Vector3::new(1f32, 1f32, 1f32), Zero::zero());
        let vertex_norms = vertex_norms
            .into_iter()
            .map(|x| x.normalize())
            .collect::<Vec<_>>();
        for f in faces.into_iter() {
            let shape = Triangle::new(
                [vertices[f[0]], vertices[f[1]], vertices[f[2]]],
                [vertex_norms[f[0]], vertex_norms[f[1]], vertex_norms[f[2]]],
                [Zero::zero(), Zero::zero(), Zero::zero()],
            );
            scene.add(Arc::new(shape), material.clone());
        }
        scene.add(
            Arc::new(Sphere::new(
                Point3::new(0f32, 0.0333099 - 1000f32, 0f32),
                1000f32,
            )),
            Arc::new(Lambertian::new(Checker::new(
                SolidColor::new(Vector3::new(0.2f32, 0.3f32, 0.1f32)),
                SolidColor::new(Vector3::new(0.9f32, 0.9f32, 0.9f32)),
                40f32,
            ))),
        );
        scene
    }
    pub fn create_world3<G: Rng>(rng: &mut G) -> Scene<F, R>
    where
        F: From<[f32; F::LANES]> + Into<[f32; F::LANES]>,
    {
        let mut scene = Scene::new(Vector3::new(0f32, 0f32, 0f32), Zero::zero());
        scene.add(
            Arc::new(Sphere::new(Point3::new(0f32, -1000f32, 0f32), 1000f32)),
            Arc::new(Lambertian::new(Noise::new(Perlin::new(rng), 2f32, 7))),
        );
        scene.add(
            Arc::new(Sphere::new(Point3::new(0f32, 2f32, 0f32), 2f32)),
            Arc::new(Lambertian::new(ImageTexture::new(
                ImageReader::open("data/earthmap.jpg")
                    .unwrap()
                    .decode()
                    .unwrap(),
            ))),
        );
        scene.add(
            Arc::new(XYRect::new(3f32, 5f32, 1f32, 3f32, -2f32, true)),
            Arc::new(DiffuseLight::new(SolidColor::new(Vector3::new(
                4f32, 4f32, 4f32,
            )))),
        );
        scene
    }
    pub fn create_world4<G: Rng>(_rng: &mut G) -> Scene<F, R>
    where
        R: 'static,
    {
        let mut scene = Scene::new(Vector3::new(0f32, 0f32, 0f32), Zero::zero());
        scene.add(
            Arc::new(YZRect::new(0., 555., 0., 555., 555., false)),
            Arc::new(Lambertian::new(SolidColor::new(Vector3::new(
                0.12, 0.45, 0.15,
            )))),
        );
        scene.add(
            Arc::new(YZRect::new(0., 555., 0., 555., 0., true)),
            Arc::new(Lambertian::new(SolidColor::new(Vector3::new(
                0.65, 0.05, 0.05,
            )))),
        );
        let light_hittable = Arc::new(ZXRect::new(227., 332., 213., 343., 554., false));
        scene.add_important(
            light_hittable.clone(),
            light_hittable,
            Arc::new(DiffuseLight::new(SolidColor::new(Vector3::new(
                15., 15., 15.,
            )))),
        );
        let white = Arc::new(Lambertian::new(SolidColor::new(Vector3::new(
            0.73, 0.73, 0.73,
        ))));
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
            Cuboid::new(Vector3::new(130., 0., 65.), Vector3::new(295., 165., 230.))
                .into_iter()
                .map(|x| {
                    Arc::new(TransformHittable::new(
                        convert(Rotation3::from_axis_angle(
                            &UnitVector3::new_unchecked(Vector3::y()),
                            15f32.to_radians(),
                        )),
                        x,
                    )) as BoxedHittable<F, R>
                }),
            white.clone(),
        );
        scene.add_all(
            Cuboid::new(Vector3::new(265., 0., 295.), Vector3::new(430., 330., 460.)).into_iter(),
            white,
        );
        scene
    }
    pub fn new(param: SceneParam, scene: Scene<F, R>) -> Self {
        let width = param.renderer.width;
        let height = param.renderer.height;
        Self {
            renderer: Renderer::new(param.renderer, param.camera, scene),
            result: RenderResult::new(width, height),
            start: SystemTime::now(),
        }
    }
    pub fn run(&self, rng: &mut R)
    where
        F: From<[f32; F::LANES]>,
        F::SimdBool: From<[bool; F::LANES]>,
    {
        println!(
            "Iter {} +{}s",
            self.result.add(self.renderer.render(rng)),
            self.start.elapsed().unwrap().as_secs()
        )
    }
    pub fn get(&self, last: usize) -> Option<(gdk_pixbuf::Pixbuf, usize)>
    where
        F: SimdRealField<Element = f32>,
    {
        self.result.get_pixbuf(last)
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
    let mut rng = Pcg64::seed_from_u64(2);
    let param: SceneParam = serde_json::from_reader(File::open("data/scene.json")?)?;
    let width = param.renderer.width;
    let height = param.renderer.height;
    let example: Arc<Example<Float, ThreadRng>> =
        Arc::new(Example::new(param, Example::create_world4(&mut rng)));
    let (msg_tx, msg_rx) = async_channel::bounded(16);
    let (render_tx, render_rx) = mpsc::channel();

    let aspect_ratio = width as f32 / height as f32;
    let app = App::new(msg_tx.clone(), 800, 600, aspect_ratio);
    // Processes all application events received from signals
    {
        let state = example.clone();
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
                            new_pixbuf
                                .savev("../../data/output.jpeg", "jpeg", &[])
                                .unwrap();
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
        let state = example.clone();
        let render_tx = render_tx.clone();
        rayon::spawn(move || {
            state.run(&mut thread_rng());
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
