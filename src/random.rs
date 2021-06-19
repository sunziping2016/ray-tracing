use crate::simd::MySimdVector;
use nalgebra::{SimdRealField, Vector2, Vector3};
use rand::distributions::uniform::SampleRange;
use rand::Rng;
use std::f32::consts;

pub fn random_uniform<F, S, R: Rng>(range: S, rng: &mut R) -> F
where
    F: SimdRealField<Element = f32> + MySimdVector,
    S: SampleRange<f32> + Clone,
{
    let mut value = F::zero();
    for index in 0..F::LANES {
        value.replace(index, rng.gen_range(range.clone()));
    }
    value
}

pub fn random_in_unit_disk<F, R: Rng>(rng: &mut R) -> Vector2<F>
where
    F: SimdRealField<Element = f32> + MySimdVector,
{
    let r = random_uniform::<F, _, _>(0f32..=1f32, rng).simd_sqrt();
    let theta = random_uniform::<F, _, _>(0f32..(2.0f32 * consts::PI), rng);
    Vector2::new(r * theta.simd_cos(), r * theta.simd_sin())
}

pub fn random_to_sphere<F, R: Rng>(rng: &mut R, radius: F, distance_squared: F) -> Vector3<F>
where
    F: SimdRealField<Element = f32> + MySimdVector,
{
    let r1 = random_uniform::<F, _, _>(0f32..=1f32, rng);
    let r2 = random_uniform::<F, _, _>(0f32..=1f32, rng);
    let z =
        F::one() + r2 * ((F::one() - radius * radius / distance_squared).simd_sqrt() - F::one());
    let xy = (F::one() - z * z).simd_sqrt();
    let phi = F::simd_two_pi() * r1;
    let x = phi.simd_cos() * xy;
    let y = phi.simd_sin() * xy;
    Vector3::new(x, y, z)
}
