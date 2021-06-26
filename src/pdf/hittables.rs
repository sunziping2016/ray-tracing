use crate::pdf::Pdf;
use crate::BoxedSamplable;
use nalgebra::{Point3, SimdRealField, SimdValue, UnitVector3};
use rand::Rng;

#[derive(Clone)]
pub struct HittablesPdf<'a, F, R> {
    origin: Point3<F>,
    objects: &'a Vec<BoxedSamplable<F, R>>,
}

impl<'a, F, R> HittablesPdf<'a, F, R> {
    pub fn new(origin: Point3<F>, objects: &'a Vec<BoxedSamplable<F, R>>) -> Self {
        HittablesPdf { origin, objects }
    }
}

impl<'a, F, R: Rng> Pdf<F, R> for HittablesPdf<'a, F, R>
where
    F: SimdRealField<Element = f32>,
    F::SimdBool: SimdValue<Element = bool>,
{
    fn value(&self, direction: &UnitVector3<F>, rng: &mut R) -> F {
        let weight = 1f32 / self.objects.len() as f32;
        let mut sum = F::zero();
        for object in self.objects.iter() {
            sum += F::splat(weight)
                * object.value(&self.origin, direction, F::SimdBool::splat(true), rng);
        }
        sum
    }

    fn generate(&self, rng: &mut R) -> UnitVector3<F> {
        // FIXME: batch random?
        let index = rng.gen_range(0..self.objects.len());
        UnitVector3::new_normalize(self.objects[index].generate(&self.origin, rng))
    }
}
