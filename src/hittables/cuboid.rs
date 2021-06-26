use crate::hittable::aa_rect::{XYRect, YZRect, ZXRect};
use crate::hittables::ManyHittables;
use crate::{BoxedHittable, SimdBoolField, SimdF32Field};
use nalgebra::Vector3;
use rand::Rng;
use std::array::IntoIter;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Cuboid {
    p0: Vector3<f32>,
    p1: Vector3<f32>,
}

impl Cuboid {
    pub fn new(p0: Vector3<f32>, p1: Vector3<f32>) -> Self {
        Cuboid { p0, p1 }
    }
}

impl<F, R: Rng> ManyHittables<F, R> for Cuboid
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    type Item = BoxedHittable<F, R>;
    type Iter = IntoIter<Self::Item, 6>;

    fn into_hittables(self) -> Self::Iter
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        IntoIter::new([
            Arc::new(XYRect::new(
                self.p0[0], self.p1[0], self.p0[1], self.p1[1], self.p0[2], false,
            )) as BoxedHittable<F, R>,
            Arc::new(XYRect::new(
                self.p0[0], self.p1[0], self.p0[1], self.p1[1], self.p1[2], true,
            )),
            Arc::new(YZRect::new(
                self.p0[1], self.p1[1], self.p0[2], self.p1[2], self.p0[0], false,
            )),
            Arc::new(YZRect::new(
                self.p0[1], self.p1[1], self.p0[2], self.p1[2], self.p1[0], true,
            )),
            Arc::new(ZXRect::new(
                self.p0[2], self.p1[2], self.p0[0], self.p1[0], self.p0[1], false,
            )),
            Arc::new(ZXRect::new(
                self.p0[2], self.p1[2], self.p0[0], self.p1[0], self.p1[1], true,
            )),
        ])
    }
}
