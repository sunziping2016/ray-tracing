use crate::hittable::HitRecord;
use crate::material::{Material, ScatterRecord};
use crate::random::random_on_unit_sphere;
use crate::ray::Ray;
use crate::texture::Texture;
use crate::SimdF32Field;
use nalgebra::{SimdRealField, UnitVector3};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Isotropic<T> {
    albedo: T,
}

impl<T> Isotropic<T> {
    pub fn new(albedo: T) -> Self {
        Isotropic { albedo }
    }
}

impl<T, F: SimdRealField, R: Rng> Material<F, R> for Isotropic<T>
where
    F: SimdF32Field,
    T: Texture<F>,
{
    fn scatter(
        &self,
        r_in: &Ray<F>,
        hit_record: &HitRecord<F>,
        rng: &mut R,
    ) -> ScatterRecord<F, R> {
        let scattered = Ray::new(
            hit_record.p,
            UnitVector3::new_unchecked(random_on_unit_sphere(rng)),
            *r_in.time(),
            r_in.mask(),
        );
        let attenuation = self.albedo.value(&hit_record.uv, &hit_record.p);
        ScatterRecord::Specular {
            specular_ray: scattered,
            attenuation,
        }
    }
}
