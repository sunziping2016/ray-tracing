pub mod lambertian;

use crate::hittable::HitRecord;
use crate::pdf::Pdf;
use crate::ray::Ray;
use crate::{SimdBoolField, SimdF32Field};
use nalgebra::{SimdValue, Vector2, Vector3};
use std::sync::Arc;

pub struct ScatterRecord<F: SimdValue, P: Pdf<F>> {
    pub specular_ray: Ray<F>,
    pub attenuation: Vector3<F>,
    pub pdf: P,
}

pub trait Material {
    type Pdf<F>: Pdf<F>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>;

    fn emitted<F>(
        &self,
        r_in: &Ray<F>,
        hit_record: &HitRecord<F>,
        uv: &Vector2<F>,
        p: &Vector3<F>,
    ) -> Vector3<F>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>;
    fn scatter<F>(
        &self,
        r_in: &Ray<F>,
        hit_record: &HitRecord<F>,
    ) -> ScatterRecord<F, Self::Pdf<F>>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>;
    fn scattering_pdf<F>(&self, r_in: &Ray<F>, hit_record: &HitRecord<F>, scattered: &Ray<F>) -> F
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>;
}

impl<M: Material> Material for Arc<M> {
    type Pdf<F>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    = M::Pdf<F>;

    fn emitted<F>(
        &self,
        r_in: &Ray<F>,
        hit_record: &HitRecord<F>,
        uv: &Vector2<F>,
        p: &Vector3<F>,
    ) -> Vector3<F>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        (**self).emitted(r_in, hit_record, uv, p)
    }

    fn scatter<F>(&self, r_in: &Ray<F>, hit_record: &HitRecord<F>) -> ScatterRecord<F, Self::Pdf<F>>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        (**self).scatter(r_in, hit_record)
    }

    fn scattering_pdf<F>(&self, r_in: &Ray<F>, hit_record: &HitRecord<F>, scattered: &Ray<F>) -> F
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        (**self).scattering_pdf(r_in, hit_record, scattered)
    }
}
