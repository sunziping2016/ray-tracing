use nalgebra::{SimdRealField, Vector2, Vector3};
use rand::distributions::uniform::SampleRange;
use rand::Rng;
use std::f32::consts;

pub fn random_uniform<F, S, R: Rng>(range: S, rng: &mut R) -> F
where
    F: SimdRealField<Element = f32>,
    S: SampleRange<f32> + Clone,
{
    let mut value = F::zero();
    for index in 0..F::lanes() {
        value.replace(index, rng.gen_range(range.clone()));
    }
    value
}

pub fn random_in_unit_disk<F, R: Rng>(rng: &mut R) -> Vector2<F>
where
    F: SimdRealField<Element = f32>,
{
    let r = random_uniform::<F, _, _>(0f32..=1f32, rng).simd_sqrt();
    let theta = random_uniform::<F, _, _>(0f32..(2.0f32 * consts::PI), rng);
    Vector2::new(r * theta.simd_cos(), r * theta.simd_sin())
}

pub fn random_to_sphere<F, R: Rng>(rng: &mut R, radius: F, distance_squared: F) -> Vector3<F>
where
    F: SimdRealField<Element = f32>,
{
    let phi = random_uniform::<F, _, _>(0f32..(2f32 * consts::PI), rng);
    let r2 = random_uniform::<F, _, _>(0f32..=1f32, rng);
    let z =
        F::one() + r2 * ((F::one() - radius * radius / distance_squared).simd_sqrt() - F::one());
    let xy = (F::one() - z * z).simd_sqrt();
    let x = phi.simd_cos() * xy;
    let y = phi.simd_sin() * xy;
    Vector3::new(x, y, z)
}

pub fn random_on_unit_sphere<F, R: Rng>(rng: &mut R) -> Vector3<F>
where
    F: SimdRealField<Element = f32>,
{
    let z = random_uniform::<F, _, _>(-1f32..=1f32, rng);
    let theta = random_uniform::<F, _, _>(0f32..(2f32 * consts::PI), rng);
    let xy = (F::one() - z * z).simd_sqrt();
    let x = theta.simd_cos() * xy;
    let y = theta.simd_sin() * xy;
    Vector3::new(x, y, z)
}

pub fn random_in_unit_sphere<F, R: Rng>(rng: &mut R) -> Vector3<F>
where
    F: SimdRealField<Element = f32>,
{
    let theta = random_uniform::<F, _, _>(0f32..(2f32 * consts::PI), rng);
    let cos_phi = random_uniform::<F, _, _>(-1f32..=1f32, rng);
    let r = random_uniform::<F, _, _>(0f32..=1f32, rng).simd_powf(F::splat(1f32 / 3f32));
    let sin_phi = (F::one() - cos_phi * cos_phi).simd_sqrt();
    let x = r * sin_phi * theta.simd_cos();
    let y = r * sin_phi * theta.simd_sin();
    let z = r * cos_phi;
    Vector3::new(x, y, z)
}
