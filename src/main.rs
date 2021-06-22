#![allow(incomplete_features)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![feature(array_map)]

use async_channel::Sender;
use gtk::{ContainerExt, FrameExt, GtkWindowExt, ImageExt, WidgetExt};
use nalgebra::{SimdRealField, Vector3};
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng, SeedableRng};
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fs::File;
use std::future::Future;
use std::process;
use std::sync::{mpsc, Arc};
use std::time::SystemTime;
use v4ray::camera::CameraParam;
use v4ray::hittable::sphere::Sphere;
use v4ray::material::dielectric::Dielectric;
use v4ray::material::lambertian::Lambertian;
use v4ray::material::metal::Metal;
use v4ray::renderer::{RenderResult, Renderer, RendererParam};
use v4ray::scene::Scene;
use v4ray::texture::checker::Checker;
use v4ray::texture::solid_color::SolidColor;
use v4ray::{SimdBoolField, SimdF32Field};

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
        let mut scene = Scene::new(Vector3::new(1f32, 1f32, 1f32));
        scene.add(
            Arc::new(Sphere::new(Vector3::new(0f32, -1000f32, 0f32), 1000f32)),
            Arc::new(Lambertian::new(Checker::new(
                SolidColor::new(Vector3::new(0.2f32, 0.3f32, 0.1f32)),
                SolidColor::new(Vector3::new(0.9f32, 0.9f32, 0.9f32)),
            ))),
        );
        for a in -11..11 {
            for b in -11..11 {
                let center = Vector3::new(
                    a as f32 + 0.9f32 * rng.gen_range(0f32..=1f32),
                    0.2f32,
                    b as f32 + 0.9f32 * rng.gen_range(0f32..=1f32),
                );
                if (center - Vector3::new(4.0f32, 0.2f32, 0f32)).norm() > 0.9 {
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
            Arc::new(Sphere::new(Vector3::new(0f32, 1f32, 0f32), 1f32)),
            Arc::new(Dielectric::new(1.5f32)),
        );
        scene.add(
            Arc::new(Sphere::new(Vector3::new(-4f32, 1f32, 0f32), 1f32)),
            Arc::new(Lambertian::new(SolidColor::new(Vector3::new(
                0.4f32, 0.2f32, 0.1f32,
            )))),
        );
        scene.add(
            Arc::new(Sphere::new(Vector3::new(4f32, 1f32, 0f32), 1f32)),
            Arc::new(Metal::new(Vector3::new(0.7f32, 0.6f32, 0.5f32), 0.0)),
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
        self.result.get(last)
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
        Arc::new(Example::new(param, Example::create_world(&mut rng)));
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
