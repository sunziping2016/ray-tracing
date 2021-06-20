use crate::hittable::HitRecord;
use crate::material::{Material, ScatterRecord};
use crate::random::random_in_unit_sphere;
use crate::ray::Ray;
use crate::{SimdBoolField, SimdF32Field};
use nalgebra::{SimdRealField, SimdValue, UnitVector3, Vector3};
use rand::Rng;

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

pub fn reflect<F: SimdRealField<Element = f32>>(
    v: &UnitVector3<F>,
    n: &UnitVector3<F>,
) -> UnitVector3<F> {
    UnitVector3::new_unchecked(v.as_ref() - n.scale(v.dot(n) * F::splat(2.0f32)))
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
        return ScatterRecord::Specular {
            attenuation: Vector3::splat(self.albedo),
            specular_ray,
        };
    }
}
