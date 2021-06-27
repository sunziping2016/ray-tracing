use crate::hittable::aa_rect::{XYRect, YZRect, ZXRect};
use crate::hittables::ManyHittables;
use crate::{BoxedHittable, SimdBoolField, SimdF32Field, BoxedSamplable};
use nalgebra::Point3;
use rand::Rng;
use std::array::IntoIter;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Cuboid {
    p0: Point3<f32>,
    p1: Point3<f32>,
}

impl Cuboid {
    pub fn new(p0: Point3<f32>, p1: Point3<f32>) -> Self {
        Cuboid { p0, p1 }
    }
}

impl<F, R: Rng> ManyHittables<F, R> for Cuboid
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    type HitItem = BoxedHittable<F, R>;
    type SampleItem = BoxedSamplable<F, R>;
    type Iter = IntoIter<(Self::HitItem, Option<Self::SampleItem>), 6>;

    fn into_hittables(self) -> Self::Iter
    where
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        let rect1 = Arc::new(XYRect::new(
            self.p0[0], self.p1[0], self.p0[1], self.p1[1], self.p0[2], false,
        ));
        let rect2 = Arc::new(XYRect::new(
            self.p0[0], self.p1[0], self.p0[1], self.p1[1], self.p1[2], true,
        ));
        let rect3 = Arc::new(YZRect::new(
            self.p0[1], self.p1[1], self.p0[2], self.p1[2], self.p0[0], false,
        ));
        let rect4 = Arc::new(YZRect::new(
            self.p0[1], self.p1[1], self.p0[2], self.p1[2], self.p1[0], true,
        ));
        let rect5 = Arc::new(ZXRect::new(
            self.p0[2], self.p1[2], self.p0[0], self.p1[0], self.p0[1], false,
        ));
        let rect6 = Arc::new(ZXRect::new(
            self.p0[2], self.p1[2], self.p0[0], self.p1[0], self.p1[1], true,
        ));
        IntoIter::new([
            (rect1.clone(), Some(rect1)),
            (rect2.clone(), Some(rect2)),
            (rect3.clone(), Some(rect3)),
            (rect4.clone(), Some(rect4)),
            (rect5.clone(), Some(rect5)),
            (rect6.clone(), Some(rect6)),
        ])
    }
}
