use crate::bvh::aabb::AABB;
use nalgebra::{SimdBool, SimdRealField, SimdValue, UnitVector3, Vector3};

#[derive(Debug, Clone)]
pub struct Ray<F: SimdValue> {
    origin: Vector3<F>,
    direction: UnitVector3<F>,
    time: F,
    mask: F::SimdBool,
}

impl<F: SimdValue> Ray<F> {
    pub fn new(origin: Vector3<F>, direction: UnitVector3<F>, time: F, mask: F::SimdBool) -> Self {
        Ray {
            origin,
            direction,
            time,
            mask,
        }
    }
    pub fn at(&self, t: F) -> Vector3<F>
    where
        F: SimdRealField,
    {
        self.origin + self.direction.scale(t)
    }
    pub fn origin(&self) -> &Vector3<F> {
        &self.origin
    }
    pub fn direction(&self) -> &UnitVector3<F> {
        &self.direction
    }
    pub fn mask(&self) -> F::SimdBool {
        self.mask
    }
}

impl<F: SimdValue<Element = f32>> Ray<F> {
    pub fn intersects_aabb(&self, aabb: &AABB, mut t_min: F, mut t_max: F) -> F::SimdBool
    where
        F: SimdRealField,
    {
        let mut update_and_test = |min: f32, max: f32, origin: F, direction: F| -> F::SimdBool {
            let min_val = (F::splat(min) - origin) / direction;
            let max_val = (F::splat(max) - origin) / direction;
            t_min = min_val.simd_min(max_val).simd_max(t_min);
            t_max = min_val.simd_max(max_val).simd_min(t_max);
            t_min.simd_lt(t_max)
        };
        let mask = self.mask
            & update_and_test(aabb.min[0], aabb.max[0], self.origin[0], self.direction[0]);
        if mask.none() {
            return mask;
        }
        let mask =
            mask & update_and_test(aabb.min[1], aabb.max[1], self.origin[1], self.direction[1]);
        if mask.none() {
            return mask;
        }
        mask & update_and_test(aabb.min[2], aabb.max[2], self.origin[2], self.direction[2])
    }
}
