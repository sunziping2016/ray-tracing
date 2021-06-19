pub mod solid_color;

use crate::{SimdBoolField, SimdF32Field};
use auto_impl::auto_impl;
use nalgebra::{Vector2, Vector3};

#[auto_impl(&, &mut, Box, Rc, Arc)]
pub trait Texture {
    fn value<F>(&self, uv: Vector2<F>, p: Vector3<F>) -> Vector3<F>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>;
}
