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

macro_rules! prev (
    (0, $submac:ident ! ($($rest:tt)*)) => ($submac!(@, $($rest)*));
    (1, $submac:ident ! ($($rest:tt)*)) => ($submac!(0, $($rest)*));
    (2, $submac:ident ! ($($rest:tt)*)) => ($submac!(1, $($rest)*));
    (3, $submac:ident ! ($($rest:tt)*)) => ($submac!(2, $($rest)*));
    (4, $submac:ident ! ($($rest:tt)*)) => ($submac!(3, $($rest)*));
    (5, $submac:ident ! ($($rest:tt)*)) => ($submac!(4, $($rest)*));
    (6, $submac:ident ! ($($rest:tt)*)) => ($submac!(5, $($rest)*));
    (7, $submac:ident ! ($($rest:tt)*)) => ($submac!(6, $($rest)*));
    (8, $submac:ident ! ($($rest:tt)*)) => ($submac!(7, $($rest)*));
    (9, $submac:ident ! ($($rest:tt)*)) => ($submac!(8, $($rest)*));
    (10, $submac:ident ! ($($rest:tt)*)) => ($submac!(9, $($rest)*));
    (11, $submac:ident ! ($($rest:tt)*)) => ($submac!(10, $($rest)*));
    (12, $submac:ident ! ($($rest:tt)*)) => ($submac!(11, $($rest)*));
    (13, $submac:ident ! ($($rest:tt)*)) => ($submac!(12, $($rest)*));
    (14, $submac:ident ! ($($rest:tt)*)) => ($submac!(13, $($rest)*));
    (15, $submac:ident ! ($($rest:tt)*)) => ($submac!(14, $($rest)*));
    (16, $submac:ident ! ($($rest:tt)*)) => ($submac!(15, $($rest)*));
    (17, $submac:ident ! ($($rest:tt)*)) => ($submac!(16, $($rest)*));
    (18, $submac:ident ! ($($rest:tt)*)) => ($submac!(17, $($rest)*));
    (19, $submac:ident ! ($($rest:tt)*)) => ($submac!(18, $($rest)*));
    (20, $submac:ident ! ($($rest:tt)*)) => ($submac!(19, $($rest)*));
    (21, $submac:ident ! ($($rest:tt)*)) => ($submac!(20, $($rest)*));
    (22, $submac:ident ! ($($rest:tt)*)) => ($submac!(21, $($rest)*));
    (23, $submac:ident ! ($($rest:tt)*)) => ($submac!(22, $($rest)*));
    (24, $submac:ident ! ($($rest:tt)*)) => ($submac!(23, $($rest)*));
    (25, $submac:ident ! ($($rest:tt)*)) => ($submac!(24, $($rest)*));
    (26, $submac:ident ! ($($rest:tt)*)) => ($submac!(25, $($rest)*));
    (27, $submac:ident ! ($($rest:tt)*)) => ($submac!(26, $($rest)*));
    (28, $submac:ident ! ($($rest:tt)*)) => ($submac!(27, $($rest)*));
    (29, $submac:ident ! ($($rest:tt)*)) => ($submac!(28, $($rest)*));
    (30, $submac:ident ! ($($rest:tt)*)) => ($submac!(29, $($rest)*));
    (31, $submac:ident ! ($($rest:tt)*)) => ($submac!(30, $($rest)*));
    (32, $submac:ident ! ($($rest:tt)*)) => ($submac!(31, $($rest)*));
    (33, $submac:ident ! ($($rest:tt)*)) => ($submac!(32, $($rest)*));
    (34, $submac:ident ! ($($rest:tt)*)) => ($submac!(33, $($rest)*));
    (35, $submac:ident ! ($($rest:tt)*)) => ($submac!(34, $($rest)*));
    (36, $submac:ident ! ($($rest:tt)*)) => ($submac!(35, $($rest)*));
    (37, $submac:ident ! ($($rest:tt)*)) => ($submac!(36, $($rest)*));
    (38, $submac:ident ! ($($rest:tt)*)) => ($submac!(37, $($rest)*));
    (39, $submac:ident ! ($($rest:tt)*)) => ($submac!(38, $($rest)*));
    (40, $submac:ident ! ($($rest:tt)*)) => ($submac!(39, $($rest)*));
    (41, $submac:ident ! ($($rest:tt)*)) => ($submac!(40, $($rest)*));
    (42, $submac:ident ! ($($rest:tt)*)) => ($submac!(41, $($rest)*));
    (43, $submac:ident ! ($($rest:tt)*)) => ($submac!(42, $($rest)*));
    (44, $submac:ident ! ($($rest:tt)*)) => ($submac!(43, $($rest)*));
    (45, $submac:ident ! ($($rest:tt)*)) => ($submac!(44, $($rest)*));
    (46, $submac:ident ! ($($rest:tt)*)) => ($submac!(45, $($rest)*));
    (47, $submac:ident ! ($($rest:tt)*)) => ($submac!(46, $($rest)*));
    (48, $submac:ident ! ($($rest:tt)*)) => ($submac!(47, $($rest)*));
    (49, $submac:ident ! ($($rest:tt)*)) => ($submac!(48, $($rest)*));
    (50, $submac:ident ! ($($rest:tt)*)) => ($submac!(49, $($rest)*));
    (51, $submac:ident ! ($($rest:tt)*)) => ($submac!(50, $($rest)*));
    (52, $submac:ident ! ($($rest:tt)*)) => ($submac!(51, $($rest)*));
    (53, $submac:ident ! ($($rest:tt)*)) => ($submac!(52, $($rest)*));
    (54, $submac:ident ! ($($rest:tt)*)) => ($submac!(53, $($rest)*));
    (55, $submac:ident ! ($($rest:tt)*)) => ($submac!(54, $($rest)*));
    (56, $submac:ident ! ($($rest:tt)*)) => ($submac!(55, $($rest)*));
    (57, $submac:ident ! ($($rest:tt)*)) => ($submac!(56, $($rest)*));
    (58, $submac:ident ! ($($rest:tt)*)) => ($submac!(57, $($rest)*));
    (59, $submac:ident ! ($($rest:tt)*)) => ($submac!(58, $($rest)*));
    (60, $submac:ident ! ($($rest:tt)*)) => ($submac!(59, $($rest)*));
    (61, $submac:ident ! ($($rest:tt)*)) => ($submac!(60, $($rest)*));
    (62, $submac:ident ! ($($rest:tt)*)) => ($submac!(61, $($rest)*));
    (63, $submac:ident ! ($($rest:tt)*)) => ($submac!(62, $($rest)*));
);

macro_rules! slice {
    (@, $slice:ident, $($rest:expr)*) => { Self::new($($rest),*) };
    ($num:tt, $slice:ident, $($rest:expr)*) => { prev!($num, slice!($slice, $slice[$num] $($rest)*)) };
}

macro_rules! impl_from_slice_for_bool {
    ($($ty:ident $num:tt)+) => {
        $(
            impl MyFromSlice for simd::$ty {
                fn from_slice(slice: &[Self::Element]) -> Self {
                    slice!($num, slice,)
                }
            }
        )+
    };
}

impl_from_slice_for_bool!(
    m8x2 1 m8x4 3 m8x8 7 m8x16 15 m8x32 31 m8x64 63
    m16x2 1 m16x4 3 m16x8 7 m16x16 15 m16x32 31
    m32x2 1 m32x4 3 m32x8 7 m32x16 15
    m64x2 1 m64x4 3 m64x8 7
    m128x1 0 m128x2 1 m128x4 3
    msizex2 1 msizex4 3 msizex8 7
);

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
                    + MySimdVector;
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
