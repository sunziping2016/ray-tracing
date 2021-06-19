pub mod sphere;

use crate::bvh::aabb::AABB;
use crate::ray::Ray;
use crate::{SimdBoolField, SimdF32Field};
use auto_impl::auto_impl;
use nalgebra::{ClosedAdd, Scalar, SimdBool, SimdValue, UnitVector3, Vector2, Vector3};
use num_traits::Zero;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct HitRecord<F: SimdValue> {
    pub p: Vector3<F>,
    pub normal: UnitVector3<F>,
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
            normal: UnitVector3::new_unchecked(Vector3::splat(val.normal.into_inner())),
            t: F::splat(val.t),
            uv: Vector2::splat(val.uv),
            front_face: F::SimdBool::splat(val.front_face),
            mask: F::SimdBool::splat(val.mask),
        }
    }
    fn extract(&self, i: usize) -> Self::Element {
        HitRecord {
            p: self.p.extract(i),
            normal: UnitVector3::new_unchecked(self.normal.as_ref().extract(i)),
            t: self.t.extract(i),
            uv: self.uv.extract(i),
            front_face: self.front_face.extract(i),
            mask: self.mask.extract(i),
        }
    }
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        HitRecord {
            p: self.p.extract_unchecked(i),
            normal: UnitVector3::new_unchecked(self.normal.as_ref().extract_unchecked(i)),
            t: self.t.extract_unchecked(i),
            uv: self.uv.extract_unchecked(i),
            front_face: self.front_face.extract_unchecked(i),
            mask: self.mask.extract_unchecked(i),
        }
    }
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.p.replace(i, val.p);
        self.normal
            .as_mut_unchecked()
            .replace(i, val.normal.into_inner());
        self.t.replace(i, val.t);
        self.uv.replace(i, val.uv);
        self.front_face.replace(i, val.front_face);
        self.mask.replace(i, val.mask);
    }
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.p.replace_unchecked(i, val.p);
        self.normal
            .as_mut_unchecked()
            .replace_unchecked(i, val.normal.into_inner());
        self.t.replace_unchecked(i, val.t);
        self.uv.replace_unchecked(i, val.uv);
        self.front_face.replace_unchecked(i, val.front_face);
        self.mask.replace_unchecked(i, val.mask);
    }

    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        Self {
            p: self.p.select(cond, other.p),
            normal: UnitVector3::new_unchecked(
                self.normal
                    .into_inner()
                    .select(cond, other.normal.into_inner()),
            ),
            t: self.t.select(cond, other.t),
            uv: self.uv.select(cond, other.uv),
            front_face: self.front_face.select(cond, other.front_face),
            mask: self.mask.select(cond, other.mask),
        }
    }
}

impl<F: SimdF32Field> HitRecord<F> {
    pub fn face_normal(
        direction: &UnitVector3<F>,
        outward_normal: UnitVector3<F>,
    ) -> (F::SimdBool, UnitVector3<F>) {
        let front_face = direction.dot(&outward_normal).is_simd_negative();
        let outward_normal = outward_normal.into_inner();
        let normal = front_face.if_else(|| outward_normal, || -outward_normal);
        // let normal = Vector3::new(
        //     front_face.if_else(|| outward_normal[0], || -outward_normal[0]),
        //     front_face.if_else(|| outward_normal[1], || -outward_normal[1]),
        //     front_face.if_else(|| outward_normal[2], || -outward_normal[2]),
        // );
        (front_face, UnitVector3::new_unchecked(normal))
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
            normal: UnitVector3::new_unchecked(Vector3::splat(Vector3::y())),
            t: F::splat(f32::INFINITY),
            uv: Zero::zero(),
            front_face: F::SimdBool::splat(false),
            mask: F::SimdBool::splat(false),
        }
    }
}
#[auto_impl(&, &mut, Box, Rc, Arc)]
pub trait Hittable {
    fn bounding_box(&self, time0: f32, time1: f32) -> AABB;
    fn hit<F>(&self, ray: &Ray<F>, t_min: F, t_max: F) -> HitRecord<F>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>;
    fn pdf_value<F>(&self, origin: &Vector3<F>, direction: &Vector3<F>) -> F
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>;
    fn random<F, R: Rng>(&self, rng: &mut R, origin: &Vector3<F>) -> Vector3<F>
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>;
}