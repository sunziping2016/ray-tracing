use crate::bvh::bvh::BVH;
use crate::camera::{Camera, CameraParam};
use crate::hittable::HitRecord;
use crate::material::ScatterRecord;
use crate::py::{PyRng, PySimd};
use crate::ray::Ray;
use crate::scene::{PyScene, Scene};
use crate::simd::MySimdVector;
use crate::{extract, EPSILON};
use crate::{SimdBoolField, SimdF32Field};
use arrayvec::ArrayVec;
use itertools::iproduct;
use nalgebra::{SimdRealField, SimdValue, Vector2, Vector3};
use num_traits::{cast, clamp};
use numpy::PyArray;
use pyo3::proc_macro::{pyclass, pymethods};
use pyo3::{IntoPy, PyObject, PyResult, Python};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use std::{array, iter};

#[pyclass(name = "RendererParam")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RendererParam {
    pub width: u32,
    pub height: u32,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub max_depth: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub antialias: Option<bool>,
}

#[pymethods]
impl RendererParam {
    #[new]
    pub fn py_new(
        width: u32,
        height: u32,
        max_depth: Option<u32>,
        antialias: Option<bool>,
    ) -> Self {
        Self {
            width,
            height,
            max_depth,
            antialias,
        }
    }
}

pub struct Renderer<F, R: Rng> {
    param: RendererParam,
    camera: Camera,
    scene: Scene<F, R>,
    bvh: BVH,
}

impl<F, R: Rng> Renderer<F, R>
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    pub fn new(param: RendererParam, camera: CameraParam, scene: Scene<F, R>) -> Self {
        let default_aspect_ratio = param.width as f32 / param.height as f32;
        let bvh = scene.build_bvh(camera.time0.unwrap_or(0f32), camera.time1.unwrap_or(0f32));
        Self {
            param,
            camera: Camera::new(camera, default_aspect_ratio),
            scene,
            bvh,
        }
    }
    fn sample(&self, rng: &mut R) -> Vec<(Vector2<F>, F::SimdBool)>
    where
        F: From<[f32; F::LANES]>,
    {
        let width = self.param.width as f32;
        let height = self.param.height as f32;
        let (xs, ys): (Vec<_>, Vec<_>) =
            iproduct!((0..self.param.height).rev(), (0..self.param.width).rev())
                .map(|(j, i)| {
                    if let Some(false) = self.param.antialias {
                        (i as f32 / width, j as f32 / height)
                    } else {
                        (
                            rng.gen_range(((i as f32 - 0.5) / width)..((i as f32 + 0.5) / width)),
                            rng.gen_range(((j as f32 - 0.5) / height)..((j as f32 + 0.5) / height)),
                        )
                    }
                })
                .chain(iter::repeat((f32::NAN, f32::NAN)).take(F::lanes() - 1))
                .unzip();
        xs.chunks_exact(F::lanes())
            .map(|chunk| F::from(extract!(chunk, |x| *x)))
            .zip(
                ys.chunks_exact(F::lanes())
                    .map(|chunk| F::from(extract!(chunk, |x| *x))),
            )
            .map(|(x, y)| (Vector2::new(x, y), x.simd_eq(x)))
            .collect()
    }
    fn ray_color(&self, rays: Vec<Ray<F>>, depth: u32, rng: &mut R) -> Vec<Vector3<F>>
    where
        F: From<[f32; F::LANES]>,
        F::SimdBool: From<[bool; F::LANES]>,
    {
        if depth == 0 {
            return vec![Vector3::splat(self.scene.environment()); rays.len()];
        }
        let mut hit_shapes: Vec<(Vec<usize>, Vec<Ray<f32>>)> =
            vec![(Vec::new(), Vec::new()); self.scene.len()];
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
                let shape = &self.scene.hittable(shape_index);
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
        let mut colors = vec![Vector3::splat(self.scene.background()); rays.len()];
        // println!(
        //     "depth: {}  hit rate: {}",
        //     depth,
        //     hit_ray_indices.len() as f32 / F::LANES as f32 / rays.len() as f32
        // );
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
                let material = &self.scene.material(shape_index);
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
    pub fn render(&self, rng: &mut R) -> Vec<Vector3<F>>
    where
        F: From<[f32; F::LANES]>,
        F::SimdBool: From<[bool; F::LANES]>,
    {
        let rays = self
            .sample(rng)
            .into_iter()
            .map(|(st, mask)| self.camera.get_ray(st, mask, rng))
            .collect::<Vec<_>>();
        self.ray_color(rays, self.param.max_depth.unwrap_or(20), rng)
    }
}

#[derive(Debug)]
pub struct RenderResult<F> {
    width: u32,
    height: u32,
    result: RwLock<(Vec<Vector3<F>>, usize)>,
}

impl<F> RenderResult<F> {
    pub fn new(width: u32, height: u32) -> Self
    where
        F: SimdRealField,
    {
        let num_pixels = (width * height) as usize;
        let num_soa = (num_pixels + F::lanes() - 1) / F::lanes();
        Self {
            width,
            height,
            result: RwLock::new((vec![Vector3::from_element(F::zero()); num_soa], 0)),
        }
    }
    pub fn add(&self, colors: Vec<Vector3<F>>) -> usize
    where
        F: SimdRealField,
    {
        let mut lock = self.result.write().unwrap();
        lock.0
            .iter_mut()
            .zip(colors.into_iter())
            .for_each(|(sum, v)| {
                *sum += v;
            });
        lock.1 += 1;
        lock.1
    }
    #[allow(clippy::needless_collect)]
    pub fn get(&self, last: usize) -> Option<(gdk_pixbuf::Pixbuf, usize)>
    where
        F: SimdRealField<Element = f32>,
    {
        let lock = self.result.read().unwrap();
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
        let height = self.height;
        let width = self.width;
        let mut bytes: Vec<u8> = vec![255; (height * width) as usize * 3];
        let min: <F as SimdValue>::Element = cast(0.5).unwrap();
        let max: <F as SimdValue>::Element = cast(255.5).unwrap();
        iproduct!(0..height, 0..width).for_each(|(y, x)| {
            let index = (y * width + x) as usize;
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

#[pyclass(name = "Renderer")]
pub struct PyRenderer {
    inner: Arc<Renderer<PySimd, PyRng>>,
}

#[pymethods]
impl PyRenderer {
    #[new]
    fn py_new(param: &RendererParam, camera: &CameraParam, scene: &PyScene) -> Self {
        Self {
            inner: Arc::new(Renderer::new(
                param.clone(),
                camera.clone(),
                scene.inner.clone(),
            )),
        }
    }
    #[name = "render"]
    fn py_render(&self, py: Python) -> PyResult<PyObject> {
        let (tx, rx) = async_channel::bounded(1);
        let inner = self.inner.clone();
        let width = self.inner.param.width as usize;
        let height = self.inner.param.height as usize;
        rayon::spawn(move || {
            let result = inner.render(&mut thread_rng());
            let _ = futures::executor::block_on(
                tx.send(
                    iproduct!(0..(height * width), 0..3)
                        .map(|(index, channel)| {
                            result[index / PySimd::LANES][channel].extract(index % PySimd::LANES)
                        })
                        .collect::<Vec<_>>(),
                ),
            );
        });
        pyo3_asyncio::async_std::into_coroutine(py, async move {
            let result = rx.recv().await.unwrap();
            Ok(Python::with_gil(move |py| {
                PyArray::from_vec(py, result)
                    .reshape((height, width, 3))
                    .unwrap()
                    .into_py(py)
            }))
        })
    }
}
