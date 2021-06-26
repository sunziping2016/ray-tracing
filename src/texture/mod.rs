pub mod checker;
pub mod image;
pub mod noise;
pub mod py;
pub mod solid_color;

use auto_impl::auto_impl;
use nalgebra::{Point3, Vector2, Vector3};
use pyo3::{Py, PyClass, Python};

#[auto_impl(&, &mut, Box, Rc, Arc)]
pub trait Texture<F> {
    fn value(&self, uv: &Vector2<F>, p: &Point3<F>) -> Vector3<F>;
}

impl<T, F> Texture<F> for Py<T>
where
    T: Texture<F> + PyClass,
{
    fn value(&self, uv: &Vector2<F>, p: &Point3<F>) -> Vector3<F> {
        Python::with_gil(|py| self.borrow(py).value(uv, p))
    }
}
