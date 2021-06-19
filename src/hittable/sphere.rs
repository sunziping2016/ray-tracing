use crate::bvh::aabb::AABB;
use crate::hittable::{HitRecord, Hittable};
use crate::random::random_to_sphere;
use crate::ray::Ray;
use crate::{SimdBoolField, SimdF32Field};
use nalgebra::{Rotation3, SimdBool, SimdRealField, SimdValue, UnitVector3, Vector2, Vector3};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Sphere {
    center: Vector3<f32>,
    radius: f32,
}

impl Sphere {
    pub fn new(center: Vector3<f32>, radius: f32) -> Self {
        Sphere { center, radius }
    }
    pub fn sphere_uv<F>(p: &Vector3<F>) -> (F, F)
    where
        F: SimdRealField,
    {
        let theta = -p[1].simd_acos();
        let phi = (-p[2]).simd_atan2(p[1]) + F::simd_pi();
        (phi / F::simd_two_pi(), theta * F::simd_frac_1_pi())
    }
}

impl Hittable for Sphere {
    fn bounding_box(&self, _time0: f32, _time1: f32) -> AABB {
        let radius = Vector3::from_element(self.radius);
        AABB::with_bounds(self.center - radius, self.center + radius)
    }
    #[allow(clippy::many_single_char_names)]
    fn hit<F>(&self, ray: &Ray<F>, t_min: F, t_max: F) -> HitRecord<F>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        let center = Vector3::splat(self.center);
        let oc: Vector3<F> = ray.origin() - center;
        let half_b = oc.dot(ray.direction());
        let c = oc.norm_squared() - F::splat(self.radius * self.radius);
        let discriminant = half_b * half_b - c;
        let mask = ray.mask() & !discriminant.is_simd_negative();
        if mask.none() {
            return Default::default();
        }
        let sqrt_d = discriminant.simd_sqrt();
        let root1 = -half_b - sqrt_d;
        let mask1 = mask & root1.simd_ge(t_min) & root1.simd_le(t_max);
        let root2 = -half_b + sqrt_d;
        let mask2 = mask & root2.simd_ge(t_min) & root2.simd_le(t_max);
        let mask = mask1 | mask2;
        if mask.none() {
            return Default::default();
        }
        let t = mask1.if_else(|| root1, || root2);
        let p = ray.at(t);
        let outward_normal = UnitVector3::new_normalize(p - center);
        let (front_face, normal) = HitRecord::face_normal(ray.direction(), outward_normal);
        let (u, v) = Self::sphere_uv(outward_normal.as_ref());
        HitRecord {
            p,
            normal,
            t,
            uv: Vector2::new(u, v),
            front_face,
            mask,
        }
    }
    fn pdf_value<F>(&self, origin: &Vector3<F>, _direction: &Vector3<F>) -> F
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        let cos_theta_max = (F::one()
            - F::splat(self.radius * self.radius)
                / (Vector3::splat(self.center) - origin).norm_squared())
        .simd_sqrt();
        let solid_angle = F::simd_two_pi() * (F::one() - cos_theta_max);
        solid_angle.simd_recip()
    }
    fn random<F, R: Rng>(&self, rng: &mut R, origin: &Vector3<F>) -> Vector3<F>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        let direction = Vector3::splat(self.center) - origin;
        let selector = direction.normalize()[0].simd_abs().simd_gt(F::splat(0.9));
        let up = Vector3::new(
            selector.if_else(F::zero, F::one),
            selector.if_else(F::one, F::zero),
            F::zero(),
        );
        let rot = Rotation3::face_towards(&direction, &up);
        rot * random_to_sphere(rng, F::splat(self.radius), direction.norm_squared())
    }
}
