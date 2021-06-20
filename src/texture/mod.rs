pub mod solid_color;

use auto_impl::auto_impl;
use nalgebra::{Vector2, Vector3};

#[auto_impl(&, &mut, Box, Rc, Arc)]
pub trait Texture<F> {
    fn value(&self, uv: Vector2<F>, p: Vector3<F>) -> Vector3<F>;
}
