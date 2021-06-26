use crate::pdf::Pdf;
use crate::random::random_uniform;
use crate::SimdF32Field;
use nalgebra::{Rotation3, Scalar, SimdBool, UnitVector3, Vector3};
use rand::Rng;

pub struct CosinePdf<F>
where
    F: Scalar,
{
    dir: UnitVector3<F>,
    rot: Rotation3<F>,
}

impl<F: SimdF32Field> CosinePdf<F> {
    pub fn new(dir: UnitVector3<F>) -> Self {
        let selector = dir[0].simd_abs().simd_gt(F::splat(0.9));
        let up = Vector3::new(
            selector.if_else(F::zero, F::one),
            selector.if_else(F::one, F::zero),
            F::zero(),
        );
        let rot = Rotation3::face_towards(&dir, &up);
        Self { dir, rot }
    }
}

impl<F, R: Rng> Pdf<F, R> for CosinePdf<F>
where
    F: SimdF32Field,
{
    fn value(&self, direction: &UnitVector3<F>, _rng: &mut R) -> F {
        let cosine = direction.as_ref().dot(&self.dir);
        cosine
            .is_simd_positive()
            .if_else(|| cosine * F::simd_frac_1_pi(), F::zero)
    }
    fn generate(&self, rng: &mut R) -> UnitVector3<F> {
        let r1 = random_uniform(0f32..1f32, rng);
        let r2 = random_uniform(0f32..1f32, rng);
        let z = (F::one() - r2).simd_sqrt();
        let phi = F::simd_two_pi() * r1;
        let sqrt_r2 = r2.simd_sqrt();
        let x = phi.simd_cos() * sqrt_r2;
        let y = phi.simd_sin() * sqrt_r2;
        UnitVector3::new_unchecked(self.rot * Vector3::new(x, y, z))
    }
}
