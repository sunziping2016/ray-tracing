use async_channel::Sender;
use gtk::{ContainerExt, FrameExt, GtkWindowExt, ImageExt, WidgetExt};
use itertools::iproduct;
use nalgebra::{SimdBool, SimdRealField, SimdValue, Vector3};
use num_traits::Zero;
use num_traits::{cast, clamp, NumCast};
use rand::thread_rng;
use ray_tracing::camera::{Camera, CameraParam};
use ray_tracing::image::ImageParam;
use ray_tracing::simd_f32::SimdF32Field;
use serde::{Deserialize, Serialize};
use simba::simd::f32x16;
use std::fmt::Debug;
use std::fs::File;
use std::future::Future;
use std::marker::PhantomData;
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
pub struct SceneParam<R> {
    image: ImageParam,
    camera: CameraParam,
    #[serde(skip_serializing, default)]
    phantom: PhantomData<R>,
}

#[derive(Debug)]
pub struct Render<R> {
    scene: SceneParam<R>,
    camera: Camera,
    screen: RwLock<(Vec<Vector3<R>>, usize)>,
}

impl<R> Render<R> {
    pub fn new(scene: SceneParam<R>) -> Self
    where
        R: SimdRealField,
    {
        let num_pixels = scene.image.height() * scene.image.width();
        let num_soa = (num_pixels + R::lanes() - 1) / R::lanes();
        let default_aspect_ratio = scene.image.width() as f32 / scene.image.height() as f32;
        let camera_param = scene.camera.clone();
        Self {
            scene,
            camera: Camera::new(camera_param, default_aspect_ratio),
            screen: RwLock::new((vec![Zero::zero(); num_soa], 0)),
        }
    }
    pub fn run(&self)
    where
        R: SimdF32Field,
        <R as SimdValue>::Element: Copy + PartialEq + Debug,
    {
        let mut rng = thread_rng();
        let result = self
            .scene
            .image
            .sample::<R, _>(&mut rng)
            .into_iter()
            .map(|(pos, mask)| {
                (
                    mask.if_else(
                        || {
                            Vector3::new(
                                R::from_subset(&0.1) + R::from_subset(&0.8) * pos[0],
                                R::from_subset(&0.9) - R::from_subset(&0.5) * pos[1],
                                R::from_subset(&1.0),
                            )
                        },
                        Zero::zero,
                    ),
                    mask,
                )
            })
            .collect::<Vec<_>>();
        let mut lock = self.screen.write().unwrap();
        lock.0
            .iter_mut()
            .zip(result.iter().copied())
            .for_each(|(sum, (v, mask))| {
                *sum += mask.if_else(|| v, Zero::zero);
            });
        lock.1 += 1;
        println!("Iter {}", lock.1);
        drop(lock)
    }
    pub fn get(&self, last: usize) -> Option<(gdk_pixbuf::Pixbuf, usize)>
    where
        R: SimdRealField,
        <R as SimdValue>::Element: Copy + PartialOrd + NumCast,
    {
        let lock = self.screen.read().unwrap();
        let new_last = lock.1;
        if new_last <= last {
            return None;
        }
        let scale = R::from_subset(&(256.0 / lock.1 as f64));
        let colors = lock.0.iter().map(|x| x.scale(scale)).collect::<Vec<_>>();
        drop(lock);
        let height = self.scene.image.height();
        let width = self.scene.image.width();
        let mut bytes: Vec<u8> = vec![255; height * width * 3];
        let min: <R as SimdValue>::Element = cast(0.5).unwrap();
        let max: <R as SimdValue>::Element = cast(255.5).unwrap();
        iproduct!(0..height, 0..width).for_each(|(y, x)| {
            let index = y * width + x;
            let (index1, index2) = (index / f32x16::lanes(), index % f32x16::lanes());
            let base = index * 3;
            bytes[base] = cast(clamp(colors[index1][0].extract(index2), min, max)).unwrap();
            bytes[base + 1] = cast(clamp(colors[index1][1].extract(index2), min, max)).unwrap();
            bytes[base + 2] = cast(clamp(colors[index1][2].extract(index2), min, max)).unwrap();
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

    let state: Arc<Render<f32x16>> = Arc::new(Render::new(serde_json::from_reader(File::open(
        "data/scene.json",
    )?)?));
    let (msg_tx, msg_rx) = async_channel::bounded(16);
    let (render_tx, render_rx) = mpsc::channel();

    let aspect_ratio = state.scene.image.width() as f32 / state.scene.image.height() as f32;
    let app = App::new(msg_tx.clone(), 400, 225, aspect_ratio);
    // Processes all application events received from signals
    {
        let state = state.clone();
        glib::MainContext::default().spawn_local(async move {
            let mut pixbuf: Option<gdk_pixbuf::Pixbuf> = None;
            // let num_pixels = config.image.height() * config.image.width();
            // let zeros: Vector3<f32x16> = Zero::zero();
            // let num_soa = (num_pixels + f32x16::lanes() - 1) / f32x16::lanes();
            // let mut sum: Vec<Vector3<f32x16>> = vec![zeros; num_soa];
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
                                // app.image.set_size_request(0, 0);
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
                            // app.image.set_size_request(0, 0);
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
        let _ = glib::source::timeout_add(1000 / 60, move || {
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
