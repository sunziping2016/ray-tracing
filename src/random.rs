use crate::simd::MySimdVector;
use nalgebra::{SimdRealField, Vector2};
use rand::distributions::uniform::SampleRange;
use rand::Rng;
use std::f32::consts;

pub fn random_uniform<F, S, R: Rng>(range: S, rng: &mut R) -> F
where
    F: SimdRealField<Element = f32> + MySimdVector + From<[f32; F::LANES]>,
    S: SampleRange<f32> + Clone,
{
    let mut indices = [0f32; F::LANES];
    for index in indices.iter_mut() {
        *index = rng.gen_range(range.clone());
    }
    indices.into()
}

pub fn random_in_unit_disk<F, R: Rng>(rng: &mut R) -> Vector2<F>
where
    F: SimdRealField<Element = f32> + MySimdVector + From<[f32; F::LANES]>,
{
    let r = random_uniform::<F, _, _>(0f32..=1f32, rng).simd_sqrt();
    let theta = random_uniform::<F, _, _>(0f32..(2.0f32 * consts::PI), rng);
    Vector2::new(r * theta.simd_cos(), r * theta.simd_sin())
}
