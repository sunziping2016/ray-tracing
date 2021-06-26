use crate::pdf::Pdf;
use crate::BoxedHittable;
use nalgebra::{SimdRealField, SimdValue, UnitVector3, Vector3};
use rand::Rng;

#[derive(Clone)]
pub struct HittablesPdf<'a, F, R> {
    origin: Vector3<F>,
    objects: &'a Vec<BoxedHittable<F, R>>,
}

impl<'a, F, R> HittablesPdf<'a, F, R> {
    pub fn new(origin: Vector3<F>, objects: &'a Vec<BoxedHittable<F, R>>) -> Self {
        HittablesPdf { origin, objects }
    }
}

impl<'a, F, R: Rng> Pdf<F, R> for HittablesPdf<'a, F, R>
where
    F: SimdRealField<Element = f32>,
    F::SimdBool: SimdValue<Element = bool>,
{
    fn value(&self, direction: &UnitVector3<F>) -> F {
        let weight = 1f32 / self.objects.len() as f32;
        let mut sum = F::zero();
        for object in self.objects.iter() {
            sum += F::splat(weight)
                * object.pdf_value(&self.origin, direction, F::SimdBool::splat(true));
        }
        sum
    }

    fn generate(&self, rng: &mut R) -> UnitVector3<F> {
        // FIXME: batch random?
        let index = rng.gen_range(0..self.objects.len());
        UnitVector3::new_normalize(self.objects[index].random(rng, &self.origin))
    }
}
