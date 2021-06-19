use crate::hittable::HitRecord;
use crate::material::{Material, ScatterRecord};
use crate::pdf::cosine::CosinePdf;
use crate::ray::Ray;
use crate::texture::Texture;
use crate::{SimdBoolField, SimdF32Field};
use nalgebra::{SimdBool, Vector2, Vector3};
use num_traits::Zero;

#[derive(Debug, Clone)]
pub struct Lambertian<T: Texture> {
    texture: T,
}

impl<T: Texture> Lambertian<T> {
    pub fn new(texture: T) -> Self {
        Lambertian { texture }
    }
}

impl<T: Texture> Material for Lambertian<T> {
    type Pdf<F>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    = CosinePdf<F>;

    fn emitted<F>(&self, _uv: &Vector2<F>, _p: &Vector3<F>) -> Vector3<F>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        Zero::zero()
    }
    fn scatter<F>(
        &self,
        _r_in: &Ray<F>,
        hit_record: &HitRecord<F>,
    ) -> Option<ScatterRecord<F, Self::Pdf<F>>>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        Some(ScatterRecord {
            specular_ray: Ray::default(),
            attenuation: self.texture.value(hit_record.uv, hit_record.p),
            pdf: CosinePdf::new(hit_record.normal),
        })
    }
    fn scattering_pdf<F>(&self, _r_in: &Ray<F>, hit_record: &HitRecord<F>, scattered: &Ray<F>) -> F
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        let cosine = hit_record.normal.dot(scattered.direction());
        cosine
            .is_simd_negative()
            .if_else(|| F::zero(), || cosine / F::simd_pi())
    }
}
