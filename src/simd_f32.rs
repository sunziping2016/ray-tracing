use nalgebra::SimdRealField;
use simba::simd;

pub trait SimdF32Field: SimdRealField {
    fn from_f32_slice(slice: &[f32]) -> Self;
    fn is_nan(self) -> Self::SimdBool;
}

macro_rules! impl_simd_f32_field {
    ($($type:ty),*) => {
        $(
            impl SimdF32Field for $type {
                fn from_f32_slice(slice: &[f32]) -> Self {
                    Self::from_slice_unaligned(slice)
                }
                fn is_nan(self) -> Self::SimdBool {
                    simd::Simd(self.0.is_nan())
                }
            }
        )*
    };
}

impl SimdF32Field for f32 {
    fn from_f32_slice(slice: &[f32]) -> Self {
        slice[0]
    }
    fn is_nan(self) -> Self::SimdBool {
        self.is_nan()
    }
}

impl_simd_f32_field!(simd::f32x2, simd::f32x4, simd::f32x8, simd::f32x16);
