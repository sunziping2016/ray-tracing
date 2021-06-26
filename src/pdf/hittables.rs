use crate::hittable::Samplable;
use crate::pdf::Pdf;
use nalgebra::{Point3, SimdRealField, SimdValue, UnitVector3};
use rand::Rng;

#[derive(Clone)]
pub struct HittablePdf<'a, F, O> {
    origin: Point3<F>,
    object: &'a O,
}

impl<'a, F, O> HittablePdf<'a, F, O> {
    pub fn new(origin: Point3<F>, object: &'a O) -> Self {
        HittablePdf { origin, object }
    }
}

impl<'a, F, R: Rng, O> Pdf<F, R> for HittablePdf<'a, F, O>
where
    O: Samplable<F, R>,
    F: SimdRealField<Element = f32>,
    F::SimdBool: SimdValue<Element = bool>,
{
    fn value(&self, direction: &UnitVector3<F>, mask: F::SimdBool, rng: &mut R) -> F {
        self.object.value(&self.origin, direction, mask, rng)
    }

    fn generate(&self, rng: &mut R) -> UnitVector3<F> {
        self.object.generate(&self.origin, rng)
    }
}
