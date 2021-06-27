use crate::hittable::{Hittable, Samplable};
use nalgebra::SimdRealField;
use rand::Rng;

pub mod cuboid;
pub mod group;
pub mod obj;
pub mod transform;

// There are two kinds of hittable set:
// - ManyHittables
// - GroupedHittables

pub trait ManyHittables<F: SimdRealField, R: Rng> {
    type HitItem: Hittable<F, R>;
    type SampleItem: Samplable<F, R>;
    type Iter: Iterator<Item = (Self::HitItem, Option<Self::SampleItem>)>;

    fn into_hittables(self) -> Self::Iter;
}
