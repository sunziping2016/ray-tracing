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

impl<F: SimdValue> SimdValue for HitRecord<F>
where
    F: Scalar,
    F::Element: Scalar,
    F::SimdBool: SimdValue<Element = bool, SimdBool = F::SimdBool>,
{
    type Element = HitRecord<F::Element>;
    type SimdBool = F::SimdBool;

    fn lanes() -> usize {
        F::lanes()
    }
    fn splat(val: Self::Element) -> Self {
        Self {
            p: Vector3::splat(val.p),
            normal: Vector3::splat(val.normal),
            t: F::splat(val.t),
            uv: Vector2::splat(val.uv),
            front_face: F::SimdBool::splat(val.front_face),
            mask: F::SimdBool::splat(val.mask),
        }
    }
    fn extract(&self, i: usize) -> Self::Element {
        HitRecord {
            p: self.p.extract(i),
            normal: self.normal.extract(i),
            t: self.t.extract(i),
            uv: self.uv.extract(i),
            front_face: self.front_face.extract(i),
            mask: self.front_face.extract(i),
        }
    }
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        HitRecord {
            p: self.p.extract_unchecked(i),
            normal: self.normal.extract_unchecked(i),
            t: self.t.extract_unchecked(i),
            uv: self.uv.extract_unchecked(i),
            front_face: self.front_face.extract_unchecked(i),
            mask: self.front_face.extract_unchecked(i),
        }
    }
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.p.replace(i, val.p);
        self.normal.replace(i, val.normal);
        self.t.replace(i, val.t);
        self.uv.replace(i, val.uv);
        self.front_face.replace(i, val.front_face);
        self.mask.replace(i, val.mask);
    }
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.p.replace_unchecked(i, val.p);
        self.normal.replace_unchecked(i, val.normal);
        self.t.replace_unchecked(i, val.t);
        self.uv.replace_unchecked(i, val.uv);
        self.front_face.replace_unchecked(i, val.front_face);
        self.mask.replace_unchecked(i, val.mask);
    }

    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        Self {
            p: self.p.select(cond, other.p),
            normal: self.normal.select(cond, other.normal),
            t: self.t.select(cond, other.t),
            uv: self.uv.select(cond, other.uv),
            front_face: self.front_face.select(cond, other.front_face),
            mask: self.mask.select(cond, other.mask),
        }
    }
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
    F: Zero + Scalar + ClosedAdd + SimdValue<Element = f32>,
    F::SimdBool: SimdValue<Element = bool>,
{
    fn default() -> Self {
        Self {
            p: Zero::zero(),
            normal: Zero::zero(),
            t: F::splat(f32::INFINITY),
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
