use crate::hittable::HitRecord;
use crate::material::{reflect, Material, ScatterRecord};
use crate::py::PyVector3;
use crate::random::random_in_unit_sphere;
use crate::ray::Ray;
use crate::{SimdBoolField, SimdF32Field};
use nalgebra::{SimdValue, UnitVector3, Vector3};
use pyo3::proc_macro::{pyclass, pymethods};
use rand::Rng;

#[pyclass(name = "Metal")]
#[derive(Debug, Clone)]
pub struct Metal {
    albedo: Vector3<f32>,
    fuzz: f32,
}

impl Metal {
    pub fn new(albedo: Vector3<f32>, fuzz: f32) -> Self {
        Metal { albedo, fuzz }
    }
}

impl<F, R: Rng> Material<F, R> for Metal
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    fn scatter(
        &self,
        r_in: &Ray<F>,
        hit_record: &HitRecord<F>,
        rng: &mut R,
    ) -> ScatterRecord<F, R> {
        let reflected = reflect(r_in.direction(), &hit_record.normal);
        let direction = UnitVector3::new_normalize(
            reflected.as_ref() + random_in_unit_sphere(rng).scale(F::splat(self.fuzz)),
        );
        let specular_ray = Ray::new(hit_record.p, direction, *r_in.time(), r_in.mask());
        ScatterRecord::Specular {
            attenuation: Vector3::splat(self.albedo),
            specular_ray,
        }
    }
}

#[pymethods]
impl Metal {
    #[new]
    pub fn py_new(albedo: PyVector3, fuzz: f32) -> Self {
        Self::new(Vector3::new(albedo.0, albedo.1, albedo.2), fuzz)
    }
}
