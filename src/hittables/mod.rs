use crate::hittable::Hittable;
use nalgebra::SimdValue;
use rand::Rng;

pub mod cuboid;
pub mod group;
pub mod transform;

// There are two kinds of hittable set:
// - ManyHittables
// - GroupedHittables

pub trait ManyHittables<F: SimdValue, R: Rng> {
    type Item: Hittable<F, R>;
    type Iter: Iterator<Item = Self::Item>;

    fn into_hittables(self) -> Self::Iter;
}
