use crate::hittable::transform::TransformHittable;
use crate::hittables::ManyHittables;
use crate::SimdF32Field;
use nalgebra::Projective3;
use rand::Rng;
use std::iter;

#[derive(Debug, Clone)]
pub struct TransformHittables<O> {
    projective: Projective3<f32>,
    objects: O,
}

impl<O> TransformHittables<O> {
    pub fn new(projective: Projective3<f32>, objects: O) -> Self {
        TransformHittables {
            projective,
            objects,
        }
    }
}

impl<O, F: SimdF32Field, R: Rng> ManyHittables<F, R> for TransformHittables<O>
where
    O: ManyHittables<F, R>,
{
    type HitItem = TransformHittable<O::HitItem>;
    type SampleItem = TransformHittable<O::SampleItem>;
    type Iter = iter::Map<
        O::Iter,
        impl FnMut((O::HitItem, Option<O::SampleItem>)) -> (Self::HitItem, Option<Self::SampleItem>),
    >;

    fn into_hittables(self) -> Self::Iter {
        let projective = self.projective;
        self.objects.into_hittables().map(move |(x, y)| {
            (
                TransformHittable::new(projective.clone(), x),
                y.map(|x| TransformHittable::new(projective.clone(), x)),
            )
        })
    }
}
