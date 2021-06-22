use crate::bvh::aabb::AABB;
use crate::camera::CameraParam;
use crate::hittable::py::{py_init_hittable, PyHitRecord};
use crate::material::py::py_init_material;
use crate::ray::PyRay;
use crate::renderer::{PyRenderer, RendererParam};
use crate::scene::PyScene;
use crate::simd::MySimdVector;
use crate::texture::py::py_init_texture;
use crate::{BoxedHittable, BoxedMaterial, BoxedTexture};
use arrayvec::ArrayVec;
use nalgebra::SimdValue;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::proc_macro::pymodule;
use pyo3::types::PyModule;
use pyo3::{PyResult, Python};
use rand::rngs::ThreadRng;
use simba::simd::f32x8;

pub type PyVector3 = (f32, f32, f32);
pub type PySimd = f32x8;
pub type PyRng = ThreadRng;
pub type PyBoxedHittable = BoxedHittable<PySimd, PyRng>;
pub type PyBoxedMaterial = BoxedMaterial<PySimd, PyRng>;
pub type PyBoxedTexture = BoxedTexture<PySimd>;

pub fn f_to_numpy<'py, F: SimdValue<Element = f32>>(py: Python<'py>, f: &F) -> &'py PyArray1<f32> {
    PyArray1::from_iter(py, (0..F::lanes()).map(|x| f.extract(x)))
}

pub fn numpy_to_f<F: SimdValue<Element = f32> + MySimdVector>(array: &PyArray1<f32>) -> PyResult<F>
where
    F: From<[f32; F::LANES]>,
{
    Ok(F::from(
        array
            .iter()?
            .map(|x| *x)
            .collect::<ArrayVec<_, { F::LANES }>>()
            .into_inner()
            .map_err(|_| PyValueError::new_err("size mismatch"))?,
    ))
}

pub fn bits_to_numpy<F: SimdValue>(py: Python, bits: u64) -> &PyArray1<bool> {
    PyArray1::from_iter(py, (0..F::lanes()).map(|idx| (bits & (1u64 << idx)) != 0))
}

pub fn bits_to_m<F: SimdValue + MySimdVector>(bits: u64) -> F::SimdBool
where
    F::SimdBool: From<[bool; F::LANES]>,
{
    F::SimdBool::from(unsafe {
        (0..F::LANES)
            .map(|x| (1u64 << x) & bits != 0)
            .collect::<ArrayVec<_, { F::LANES }>>()
            .into_inner_unchecked()
    })
}

#[pymodule]
fn v4ray(py: Python, module: &PyModule) -> PyResult<()> {
    pyo3_asyncio::try_init(py)?;

    module.add_class::<PyScene>()?;
    module.add_class::<AABB>()?;
    module.add_class::<PyRay>()?;
    module.add_class::<PyHitRecord>()?;
    module.add_class::<CameraParam>()?;
    module.add_class::<RendererParam>()?;
    module.add_class::<PyRenderer>()?;

    let hittable = PyModule::new(py, "shape")?;
    py_init_hittable(hittable)?;
    module.add_submodule(hittable)?;

    let material = PyModule::new(py, "material")?;
    py_init_material(material)?;
    module.add_submodule(material)?;

    let texture = PyModule::new(py, "texture")?;
    py_init_texture(texture)?;
    module.add_submodule(texture)?;
    Ok(())
}
