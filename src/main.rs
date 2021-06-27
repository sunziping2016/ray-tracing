use async_std::task;
use clap::{AppSettings, Clap};
use image::{ImageBuffer, Rgb};
use rand::rngs::ThreadRng;
use rand::thread_rng;
use simba::simd::f32x8;
use std::error::Error;
use std::fs::File;
use std::sync::{mpsc, Arc};
use std::time::{Duration, SystemTime};
use v4ray::json::{build_scene, SceneParam};
use v4ray::renderer::{RenderResult, Renderer};

type F = f32x8;
type R = ThreadRng;

/// as do all doc strings on fields
#[derive(Clap)]
#[clap(version = "0.1.0", author = "Ziping Sun <me@szp.io>")]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
    #[clap(short, long, default_value = "scene.json")]
    input: String,
    #[clap(short, long, default_value = "output.bmp")]
    output: String,
}

#[async_std::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let opts: Opts = Opts::parse();
    let param: SceneParam = serde_json::from_reader(File::open(&opts.input)?)?;
    let width = param.renderer.width;
    let height = param.renderer.height;
    let scene = build_scene::<F, R>(&param);
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
            task::sleep(Duration::from_secs(5)).await;
            if let Some((bytes, new_last)) = render_result.get_raw(last) {
                if let Ok(()) = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, bytes)
                    .unwrap()
                    .save(&opts.output)
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
