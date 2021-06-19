#![allow(incomplete_features)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![feature(generic_associated_types)]

use crate::simd::MySimdVector;
use nalgebra::{SimdRealField, SimdValue};

pub mod bvh;
pub mod camera;
pub mod hittable;
pub mod image;
pub mod material;
pub mod pdf;
pub mod random;
pub mod ray;
pub mod simd;
pub mod texture;

pub const EPSILON: f32 = 0.00001;

pub trait SimdF32Field: SimdRealField<Element = f32> + MySimdVector {}

pub trait SimdBoolField<F: SimdValue>: SimdValue<Element = bool, SimdBool = F::SimdBool> {}

impl<T> SimdF32Field for T where T: SimdRealField<Element = f32> + MySimdVector {}

impl<T, F: SimdValue> SimdBoolField<F> for T where
    T: SimdValue<Element = bool, SimdBool = F::SimdBool>
{
}
