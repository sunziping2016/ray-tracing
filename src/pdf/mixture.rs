use crate::pdf::Pdf;
use crate::random::random_uniform;
use nalgebra::{SimdBool, SimdRealField, UnitVector3};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct MixturePdf<P1, P2> {
    pdf1: P1,
    pdf2: P2,
}

impl<P1, P2> MixturePdf<P1, P2> {
    pub fn new(pdf1: P1, pdf2: P2) -> Self {
        MixturePdf { pdf1, pdf2 }
    }
}

impl<F, R: Rng, P1: Pdf<F, R>, P2: Pdf<F, R>> Pdf<F, R> for MixturePdf<P1, P2>
where
    F: SimdRealField<Element = f32>,
{
    fn value(&self, direction: &UnitVector3<F>, rng: &mut R) -> F {
        F::splat(0.5f32) * self.pdf1.value(direction, rng)
            + F::splat(0.5f32) * self.pdf2.value(direction, rng)
    }

    fn generate(&self, rng: &mut R) -> UnitVector3<F> {
        let selector = random_uniform::<F, _, _>(0f32..1f32, rng).simd_lt(F::splat(0.5f32));
        let ray1 = self.pdf1.generate(rng);
        let ray2 = self.pdf2.generate(rng);
        UnitVector3::new_unchecked(ray1.zip_map(&ray2, |x, y| selector.if_else(|| x, || y)))
    }
}
