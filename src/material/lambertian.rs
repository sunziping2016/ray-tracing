use crate::hittable::HitRecord;
use crate::material::{Material, ScatterRecord};
use crate::pdf::cosine::CosinePdf;
use crate::py::{PyBoxedTexture, PyRng, PySimd};
use crate::ray::Ray;
use crate::texture::py::to_texture;
use crate::texture::Texture;
use crate::{SimdBoolField, SimdF32Field};
use nalgebra::{Vector2, Vector3};
use pyo3::proc_macro::{pyclass, pymethods};
use pyo3::{Py, PyAny, Python};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Lambertian<T> {
    texture: T,
}

impl<T> Lambertian<T> {
    pub fn new(texture: T) -> Self {
        Lambertian { texture }
    }
}

impl<F, T: Texture<F>, R: Rng> Material<F, R> for Lambertian<T>
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    fn scatter(
        &self,
        _r_in: &Ray<F>,
        hit_record: &HitRecord<F>,
        _rng: &mut R,
    ) -> ScatterRecord<F, R> {
        ScatterRecord::Scatter {
            attenuation: self.texture.value(hit_record.uv, hit_record.p),
            pdf: Box::new(CosinePdf::new(hit_record.normal)),
        }
    }
}

#[pyclass(name = "Lambertian")]
pub struct PyLambertian {
    inner: Lambertian<PyBoxedTexture>,
}

impl Material<PySimd, PyRng> for PyLambertian {
    fn emitted(&self, uv: &Vector2<PySimd>, p: &Vector3<PySimd>) -> Vector3<PySimd> {
        Material::<PySimd, PyRng>::emitted(&self.inner, uv, p)
    }

    fn scatter(
        &self,
        r_in: &Ray<PySimd>,
        hit_record: &HitRecord<PySimd>,
        rng: &mut PyRng,
    ) -> ScatterRecord<PySimd, PyRng> {
        self.inner.scatter(r_in, hit_record, rng)
    }
}

#[pymethods]
impl PyLambertian {
    #[new]
    fn py_new(py: Python, item: Py<PyAny>) -> Self {
        let texture = to_texture(py, item);
        Self {
            inner: Lambertian::new(texture),
        }
    }
}
