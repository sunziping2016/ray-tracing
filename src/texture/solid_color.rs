use crate::texture::Texture;
use nalgebra::{SimdRealField, SimdValue, Vector2, Vector3};

#[derive(Clone, Debug)]
pub struct SolidColor {
    color: Vector3<f32>,
}

impl SolidColor {
    pub fn new(color: Vector3<f32>) -> Self {
        SolidColor { color }
    }
}

impl<F> Texture<F> for SolidColor
where
    F: SimdRealField<Element = f32>,
{
    fn value(&self, _uv: Vector2<F>, _p: Vector3<F>) -> Vector3<F> {
        Vector3::splat(self.color)
    }
}
