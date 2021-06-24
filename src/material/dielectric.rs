use crate::hittable::HitRecord;
use crate::material::{reflect, refract, Material, ScatterRecord};
use crate::random::random_uniform;
use crate::ray::Ray;
use crate::SimdF32Field;
use nalgebra::{SimdBool, SimdRealField, UnitVector3, Vector3};
use pyo3::proc_macro::{pyclass, pymethods};
use rand::Rng;

#[pyclass(name = "Dielectric")]
#[derive(Debug, Clone)]
pub struct Dielectric {
    ir: f32,
}

impl Dielectric {
    pub fn new(ir: f32) -> Self {
        Dielectric { ir }
    }
}

pub fn reflectance<F: SimdRealField<Element = f32>>(cosine: F, ref_idx: F) -> F {
    let r0 = (F::one() - ref_idx) / (F::one() + ref_idx);
    let r0 = r0 * r0;
    r0 + (F::one() - r0) * (F::one() - cosine).simd_powf(F::splat(5f32))
}

impl<F, R: Rng> Material<F, R> for Dielectric
where
    F: SimdF32Field,
{
    fn scatter(
        &self,
        r_in: &Ray<F>,
        hit_record: &HitRecord<F>,
        rng: &mut R,
    ) -> ScatterRecord<F, R> {
        let refraction_ratio = hit_record
            .front_face
            .if_else(|| F::splat(self.ir).simd_recip(), || F::splat(self.ir));
        let cos_theta = -r_in.direction().dot(&hit_record.normal);
        let sin_theta = (F::one() - cos_theta * cos_theta).simd_sqrt();
        let cannot_refract = (refraction_ratio * sin_theta).simd_gt(F::one());
        let cannot_refract = cannot_refract
            | reflectance(cos_theta, refraction_ratio).simd_gt(random_uniform(0f32..1f32, rng));
        let direction = cannot_refract.if_else(
            || reflect(r_in.direction(), &hit_record.normal).into_inner(),
            || refract(r_in.direction(), &hit_record.normal, refraction_ratio).into_inner(),
        );
        ScatterRecord::Specular {
            attenuation: Vector3::from_element(F::one()),
            specular_ray: Ray::new(
                hit_record.p,
                UnitVector3::new_unchecked(direction),
                *r_in.time(),
                r_in.mask(),
            ),
        }
    }
}

#[pymethods]
impl Dielectric {
    #[new]
    pub fn py_new(ir: f32) -> Self {
        Self::new(ir)
    }
}
