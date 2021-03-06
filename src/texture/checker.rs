#[cfg(feature = "python")]
use crate::py::{PyBoxedTexture, PySimd};
#[cfg(feature = "python")]
use crate::texture::py::to_texture;
use crate::texture::Texture;
use crate::SimdF32Field;
use nalgebra::{Point3, SimdBool};
use nalgebra::{Vector2, Vector3};
#[cfg(feature = "python")]
use pyo3::proc_macro::{pyclass, pymethods};
#[cfg(feature = "python")]
use pyo3::{Py, PyAny, Python};

#[derive(Debug, Clone)]
pub struct Checker<T1, T2> {
    odd: T1,
    even: T2,
    density: f32,
}

impl<T1, T2> Checker<T1, T2> {
    pub fn new(odd: T1, even: T2, density: f32) -> Self {
        Checker { odd, even, density }
    }
}

impl<F, T1: Texture<F>, T2: Texture<F>> Texture<F> for Checker<T1, T2>
where
    F: SimdF32Field,
{
    fn value(&self, uv: &Vector2<F>, p: &Point3<F>) -> Vector3<F> {
        let density = F::splat(self.density);
        let sines =
            (density * p[0]).simd_sin() * (density * p[1]).simd_sin() * (density * p[2]).simd_sin();
        sines
            .is_simd_positive()
            .if_else(|| self.even.value(uv, p), || self.odd.value(uv, p))
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Checker")]
#[derive(Clone)]
pub struct PyChecker {
    inner: Checker<PyBoxedTexture, PyBoxedTexture>,
}

#[cfg(feature = "python")]
impl Texture<PySimd> for PyChecker {
    fn value(&self, uv: &Vector2<PySimd>, p: &Point3<PySimd>) -> Vector3<PySimd> {
        self.inner.value(uv, p)
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyChecker {
    #[new]
    pub fn py_new(py: Python, texture1: Py<PyAny>, texture2: Py<PyAny>, density: f32) -> Self {
        let texture1 = to_texture(py, texture1);
        let texture2 = to_texture(py, texture2);
        Self {
            inner: Checker::new(texture1, texture2, density),
        }
    }
}
