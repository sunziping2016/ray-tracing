use nalgebra::{Field, SimdBool, SimdPartialOrd, SimdValue};
use num_traits::{FromPrimitive, NumAssign};
use packed_simd_2::SimdVector;
use simba::simd;
use simba::simd::{PrimitiveSimdValue, Simd};
use std::fmt::Display;
use std::ops::Neg;

pub struct ConstSize<const N: usize>;

pub trait MyFromSlice: SimdValue {
    fn from_slice(slice: &[Self::Element]) -> Self;
}

macro_rules! impl_from_slice {
    ($($ty:ident)+) => {
        $(
            impl MyFromSlice for simd::$ty {
                fn from_slice(slice: &[Self::Element]) -> Self {
                    Self::from_slice_unaligned(slice)
                }
            }
        )+
    };
}

impl_from_slice!(
    // float
    f32x2 f32x4 f32x8 f32x16
    f64x2 f64x4 f64x8
    // signed
    i8x2 i8x4 i8x8 i8x16 i8x32 i8x64
    i16x2 i16x4 i16x8 i16x16 i16x32
    i32x2 i32x4 i32x8 i32x16
    i64x2 i64x4 i64x8
    i128x1 i128x2 i128x4
    isizex2 isizex4 isizex8
    // unsigned
    u8x2 u8x4 u8x8 u8x16 u8x32 u8x64
    u16x2 u16x4 u16x8 u16x16 u16x32
    u32x2 u32x4 u32x8 u32x16
    u64x2 u64x4 u64x8
    u128x1 u128x2 u128x4
    usizex2 usizex4 usizex8
);

pub trait MyFromElement: SimdValue {
    fn from_element(element: Self::Element) -> Self;
}

impl<T> MyFromElement for Simd<T>
where
    T: SimdVector + From<[Self::Element; T::LANES]>,
    Self: SimdValue,
    Self::Element: Copy,
{
    fn from_element(element: Self::Element) -> Self {
        Simd([element; T::LANES].into())
    }
}

macro_rules! impl_by_lane {
    ($tr:path { $($n:literal => $ty:ident),+ }) => {
        $(
            impl $tr for ConstSize<$n> {
                type Type = simd::$ty;
            }
        )+
    };
}

macro_rules! impl_int_by_lane {
    ($($tr:ident { $($n:literal => $ty:ident),+ })+) => {
        $(
            pub trait $tr {
                type Type: NumAssign
                    + Neg
                    + Display
                    + FromPrimitive
                    + SimdPartialOrd
                    + PrimitiveSimdValue
                    + MyFromSlice
                    + MySimdVector;
            }
            impl_by_lane!($tr { $($n => $ty),+ } );
        )+
    };
}

impl_int_by_lane!(
    I8ByLane { 2 => i8x2, 4 => i8x4, 8 => i8x8, 16 => i8x16, 32 => i8x32, 64 => i8x64 }
    I16ByLane { 2 => i16x2, 4 => i16x4, 8 => i16x8, 16 => i16x16, 32 => i16x32 }
    I32ByLane { 2 => i32x2, 4 => i32x4, 8 => i32x8, 16 => i32x16 }
    I64ByLane { 2 => i64x2, 4 => i64x4, 8 => i64x8 }
    I128ByLane { 1 => i128x1, 2 => i128x2, 4 => i128x4 }
    ISizeByLane { 2 => isizex2, 4 => isizex4, 8 => isizex8 }
);

macro_rules! impl_uint_by_lane {
    ($($tr:ident { $($n:literal => $ty:ident),+ })+) => {
        $(
            pub trait $tr {
                type Type: NumAssign
                    + Display
                    + FromPrimitive
                    + SimdPartialOrd
                    + PrimitiveSimdValue
                    + MyFromSlice
                    + MySimdVector;
            }
            impl_by_lane!($tr { $($n => $ty),+ } );
        )+
    };
}

impl_uint_by_lane!(
    U8ByLane { 2 => u8x2, 4 => u8x4, 8 => u8x8, 16 => u8x16, 32 => u8x32, 64 => u8x64 }
    U16ByLane { 2 => u16x2, 4 => u16x4, 8 => u16x8, 16 => u16x16, 32 => u16x32 }
    U32ByLane { 2 => u32x2, 4 => u32x4, 8 => u32x8, 16 => u32x16 }
    U64ByLane { 2 => u64x2, 4 => u64x4, 8 => u64x8 }
    U128ByLane { 1 => u128x1, 2 => u128x2, 4 => u128x4 }
    USizeByLane { 2 => usizex2, 4 => usizex4, 8 => usizex8 }
);

macro_rules! impl_float_by_lane {
    ($($tr:ident { $($n:literal => $ty:ident),+ })+) => {
        $(
            pub trait $tr {
                type Type: Field
                    + Display
                    + FromPrimitive
                    + PrimitiveSimdValue
                    + MyFromSlice
                    + MySimdVector;
            }
            impl_by_lane!($tr { $($n => $ty),+ } );
        )+
    };
}

impl_float_by_lane! {
    F32ByLane { 2 => f32x2, 4 => f32x4, 8 => f32x8, 16 => f32x16 }
    F64ByLane { 2 => f64x2, 4 => f64x4, 8 => f64x8 }
}

macro_rules! impl_mask_by_lane {
    ($($tr:ident { $($n:literal => $ty:ident),+ })+) => {
        $(
            pub trait $tr {
                type Type: SimdBool
                    + Display
                    + PrimitiveSimdValue
                    + MySimdVector
                    + MyMask;
            }
            impl_by_lane!($tr { $($n => $ty),+ } );
        )+
    };
}

impl_mask_by_lane!(
    M8ByLane { 2 => m8x2, 4 => m8x4, 8 => m8x8, 16 => m8x16, 32 => m8x32, 64 => m8x64 }
    M16ByLane { 2 => m16x2, 4 => m16x4, 8 => m16x8, 16 => m16x16, 32 => m16x32 }
    M32ByLane { 2 => m32x2, 4 => m32x4, 8 => m32x8, 16 => m32x16 }
    M64ByLane { 2 => m64x2, 4 => m64x4, 8 => m64x8 }
    M128ByLane { 1 => m128x1, 2 => m128x2, 4 => m128x4 }
    MSizeByLane { 2 => msizex2, 4 => msizex4, 8 => msizex8 }
);

// <ConstSize<{<T as SimdLane>::VALUE}> as I8WithLane>::Type

pub trait MySimdVector {
    type Element;
    type LanesType;

    const LANES: usize;
}

impl<T: SimdVector> MySimdVector for Simd<T> {
    type Element = T::Element;
    type LanesType = T::LanesType;
    const LANES: usize = T::LANES;
}

pub trait MyMask {
    fn zeros() -> Self;
    fn ones() -> Self;
}

macro_rules! impl_my_mask {
    ($($ty:ident)+) => {
        $(
            impl MyMask for Simd<packed_simd_2::$ty> {
                fn zeros() -> Self {
                    Simd(packed_simd_2::$ty::default())
                }
                fn ones() -> Self {
                    Simd(!packed_simd_2::$ty::default())
                }
            }
        )*
    };
}

impl_my_mask!(
    m8x2 m8x4 m8x8 m8x16 m8x32 m8x64
    m16x2 m16x4 m16x8 m16x16 m16x32
    m32x2 m32x4 m32x8 m32x16
    m64x2 m64x4 m64x8
    m128x1 m128x2 m128x4
    msizex2 msizex4 msizex8
);

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Field;

    #[allow(dead_code)]
    fn zero_mask<R: Field>() {}
}
