use crate::hittable::HitRecord;
use crate::material::Material;
use crate::texture::Texture;
use nalgebra::{SimdBool, SimdRealField, Vector3};
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
    fn emitted(&self, hit_record: &HitRecord<F>) -> Vector3<F> {
        let color = self.emit.value(&hit_record.uv, &hit_record.p);
        color.map(|x| hit_record.front_face.if_else(|| x, F::zero))
    }
}
