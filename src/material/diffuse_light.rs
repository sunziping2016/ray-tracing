use crate::material::Material;
use crate::texture::Texture;
use nalgebra::{SimdRealField, Vector2, Vector3};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct DiffuseLight<T> {
    emit: T,
}

impl<T> DiffuseLight<T> {
    pub fn new(emit: T) -> Self {
        DiffuseLight { emit }
    }
}

impl<T: Texture<F>, F: SimdRealField, R: Rng> Material<F, R> for DiffuseLight<T> {
    fn emitted(&self, uv: &Vector2<F>, p: &Vector3<F>) -> Vector3<F> {
        self.emit.value(uv, p)
    }
}
