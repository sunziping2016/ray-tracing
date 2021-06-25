pub mod dielectric;
pub mod lambertian;
pub mod metal;
pub mod py;

use crate::hittable::HitRecord;
use crate::pdf::Pdf;
use crate::ray::Ray;
use auto_impl::auto_impl;
use nalgebra::{SimdValue, UnitVector3, Vector2, Vector3};
use pyo3::{Py, PyClass, Python};
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

pub fn reflect<F: SimdRealField<Element = f32>>(
    v: &UnitVector3<F>,
    n: &UnitVector3<F>,
) -> UnitVector3<F> {
    UnitVector3::new_unchecked(v.as_ref() - n.scale(v.dot(n) * F::splat(2.0f32)))
}

pub fn refract<F: SimdRealField<Element = f32>>(
    uv: &UnitVector3<F>,
    n: &UnitVector3<F>,
    etai_over_etat: F,
) -> UnitVector3<F> {
    let cos_theta = -uv.dot(n);
    let r_out_perp = (uv.as_ref() + n.scale(cos_theta)).scale(etai_over_etat);
    let r_out_parallel = n.scale(-(F::one() - r_out_perp.norm_squared()).simd_sqrt());
    UnitVector3::new_unchecked(r_out_perp + r_out_parallel)
}

impl<T, F: SimdRealField, R: Rng> Material<F, R> for Py<T>
where
    T: Material<F, R> + PyClass,
{
    fn emitted(&self, uv: &Vector2<F>, p: &Vector3<F>) -> Vector3<F> {
        Python::with_gil(|py| self.as_ref(py).borrow().emitted(uv, p))
    }

    fn scatter(
        &self,
        r_in: &Ray<F>,
        hit_record: &HitRecord<F>,
        rng: &mut R,
    ) -> ScatterRecord<F, R> {
        Python::with_gil(|py| self.as_ref(py).borrow().scatter(r_in, hit_record, rng))
    }
}
