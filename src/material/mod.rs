pub mod lambertian;
pub mod metal;

use crate::hittable::HitRecord;
use crate::pdf::Pdf;
use crate::ray::Ray;
use auto_impl::auto_impl;
use nalgebra::{SimdValue, Vector2, Vector3};
use rand::Rng;
use simba::simd::SimdRealField;

pub enum ScatterRecord<F: SimdValue, R: Rng> {
    Specular {
        attenuation: Vector3<F>,
        specular_ray: Ray<F>,
    },
    Scatter {
        attenuation: Vector3<F>,
        pdf: Box<dyn Pdf<F, R>>,
    },
    None,
}

#[auto_impl(&, &mut, Box, Rc, Arc)]
pub trait Material<F: SimdRealField, R: Rng> {
    fn emitted(&self, _uv: &Vector2<F>, _p: &Vector3<F>) -> Vector3<F> {
        Vector3::from_element(F::zero())
    }
    fn scatter(
        &self,
        _r_in: &Ray<F>,
        _hit_record: &HitRecord<F>,
        _rng: &mut R,
    ) -> ScatterRecord<F, R> {
        ScatterRecord::None
    }
}
