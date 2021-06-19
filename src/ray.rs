use crate::bvh::aabb::AABB;
use nalgebra::{ClosedAdd, Scalar, SimdBool, SimdRealField, SimdValue, UnitVector3, Vector3};
use num_traits::{One, Zero};

#[derive(Debug, Clone)]
pub struct Ray<F: SimdValue> {
    origin: Vector3<F>,
    direction: UnitVector3<F>,
    time: F,
    mask: F::SimdBool,
}

impl<F: SimdValue> Default for Ray<F>
where
    F: Zero + Scalar + ClosedAdd,
    F::Element: Zero + One + Scalar,
    F::SimdBool: SimdValue<Element = bool>,
{
    fn default() -> Self {
        Self {
            origin: Zero::zero(),
            direction: UnitVector3::new_unchecked(Vector3::splat(Vector3::x())),
            time: Zero::zero(),
            mask: F::SimdBool::splat(false),
        }
    }
}

impl<F: SimdValue> SimdValue for Ray<F>
where
    F: Scalar,
    F::Element: Scalar,
    F::SimdBool: SimdValue<Element = bool, SimdBool = F::SimdBool>,
{
    type Element = Ray<F::Element>;
    type SimdBool = F::SimdBool;

    fn lanes() -> usize {
        F::lanes()
    }
    fn splat(val: Self::Element) -> Self {
        Self::new(
            Vector3::splat(val.origin),
            UnitVector3::new_unchecked(Vector3::splat(val.direction.into_inner())),
            F::splat(val.time),
            F::SimdBool::splat(val.mask),
        )
    }
    fn extract(&self, i: usize) -> Self::Element {
        Ray::new(
            self.origin.extract(i),
            UnitVector3::new_unchecked(Vector3::splat(self.direction.as_ref().extract(i))),
            self.time.extract(i),
            self.mask.extract(i),
        )
    }
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        Ray::new(
            self.origin.extract_unchecked(i),
            UnitVector3::new_unchecked(Vector3::splat(
                self.direction.as_ref().extract_unchecked(i),
            )),
            self.time.extract_unchecked(i),
            self.mask.extract_unchecked(i),
        )
    }
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.origin.replace(i, val.origin);
        self.direction
            .as_mut_unchecked()
            .replace(i, val.direction.into_inner());
        self.time.replace(i, val.time);
        self.mask.replace(i, val.mask);
    }
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.origin.replace_unchecked(i, val.origin);
        self.direction
            .as_mut_unchecked()
            .replace_unchecked(i, val.direction.into_inner());
        self.time.replace_unchecked(i, val.time);
        self.mask.replace_unchecked(i, val.mask);
    }
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        Self {
            origin: self.origin.select(cond, other.origin),
            direction: UnitVector3::new_unchecked(
                self.direction
                    .into_inner()
                    .select(cond, other.direction.into_inner()),
            ),
            time: self.time.select(cond, other.time),
            mask: self.mask.select(cond, other.mask),
        }
    }
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
    pub fn time(&self) -> &F {
        &self.time
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
