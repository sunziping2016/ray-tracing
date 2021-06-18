pub mod sphere;

use crate::bvh::aabb::AABB;
use crate::ray::Ray;
use nalgebra::{ClosedAdd, Scalar, SimdBool, SimdRealField, SimdValue, Vector2, Vector3};
use num_traits::Zero;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct HitRecord<F: SimdValue> {
    pub p: Vector3<F>,
    pub normal: Vector3<F>,
    // TODO: material
    pub t: F,
    pub uv: Vector2<F>,
    pub front_face: F::SimdBool,
    pub mask: F::SimdBool,
}

impl<F: SimdRealField> HitRecord<F> {
    pub fn face_normal(
        direction: Vector3<F>,
        outward_normal: Vector3<F>,
    ) -> (F::SimdBool, Vector3<F>) {
        let front_face = direction.dot(&outward_normal).is_simd_negative();
        let normal = Vector3::new(
            front_face.if_else(|| outward_normal[0], || -outward_normal[0]),
            front_face.if_else(|| outward_normal[1], || -outward_normal[1]),
            front_face.if_else(|| outward_normal[2], || -outward_normal[2]),
        );
        (front_face, normal)
    }
}

impl<F: SimdValue> Default for HitRecord<F>
where
    F: Zero + Scalar + ClosedAdd,
    F::SimdBool: SimdValue<Element = bool>,
{
    fn default() -> Self {
        Self {
            p: Zero::zero(),
            normal: Zero::zero(),
            t: Zero::zero(),
            uv: Zero::zero(),
            front_face: F::SimdBool::splat(false),
            mask: F::SimdBool::splat(false),
        }
    }
}

pub trait Bounded {
    fn bounding_box(&self, time0: f32, time1: f32) -> AABB;
}

pub trait Hittable<F: SimdValue>: Bounded {
    fn hit(&self, ray: &Ray<F>, t_min: F, t_max: F) -> HitRecord<F>;
    fn pdf_value(&self, origin: &Vector3<F>, direction: &Vector3<F>) -> F;
    fn random<R: Rng>(&self, rng: &mut R, origin: &Vector3<F>) -> Vector3<F>;
}
