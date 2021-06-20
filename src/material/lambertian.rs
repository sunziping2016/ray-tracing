use crate::hittable::HitRecord;
use crate::material::{Material, ScatterRecord};
use crate::pdf::cosine::CosinePdf;
use crate::ray::Ray;
use crate::texture::Texture;
use crate::{SimdBoolField, SimdF32Field};
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
