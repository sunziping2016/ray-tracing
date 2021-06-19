use crate::texture::Texture;
use crate::{SimdBoolField, SimdF32Field};
use nalgebra::{SimdValue, Vector2, Vector3};

#[derive(Clone, Debug)]
pub struct SolidColor {
    color: Vector3<f32>,
}

impl SolidColor {
    pub fn new(color: Vector3<f32>) -> Self {
        SolidColor { color }
    }
}

impl Texture for SolidColor {
    fn value<F>(&self, _uv: Vector2<F>, _p: Vector3<F>) -> Vector3<F>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        Vector3::splat(self.color)
    }
}
