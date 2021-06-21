use crate::texture::Texture;
use crate::SimdF32Field;
use nalgebra::SimdBool;
use nalgebra::{Vector2, Vector3};

pub struct Checker<T1, T2> {
    odd: T1,
    even: T2,
}

impl<T1, T2> Checker<T1, T2> {
    pub fn new(odd: T1, even: T2) -> Self {
        Checker { odd, even }
    }
}

impl<F, T1: Texture<F>, T2: Texture<F>> Texture<F> for Checker<T1, T2>
where
    F: SimdF32Field,
{
    fn value(&self, uv: Vector2<F>, p: Vector3<F>) -> Vector3<F> {
        let sines = (F::splat(10f32) * p[0]).simd_sin()
            * (F::splat(10f32) * p[1]).simd_sin()
            * (F::splat(10f32) * p[2]).simd_sin();
        sines
            .is_simd_positive()
            .if_else(|| self.even.value(uv, p), || self.odd.value(uv, p))
    }
}
