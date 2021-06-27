#![allow(incomplete_features)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![feature(generic_associated_types)]
#![feature(never_type)]
#![feature(associated_type_defaults)]
#![feature(array_map)]
#![feature(min_type_alias_impl_trait)]

use crate::hittable::{Hittable, Samplable};
use crate::material::Material;
use crate::simd::MySimdVector;
use crate::texture::Texture;
use nalgebra::{SimdRealField, SimdValue};
use std::fmt::Debug;
use std::sync::Arc;

pub mod bvh;
pub mod camera;
pub mod hittable;
pub mod hittables;
pub mod json;
pub mod material;
pub mod pdf;
pub mod py;
pub mod random;
pub mod ray;
pub mod renderer;
pub mod scene;
pub mod simd;
pub mod texture;

pub const EPSILON: f32 = 0.001;

pub type BoxedHittable<F, R> = Arc<dyn Hittable<F, R> + Send + Sync>;
pub type BoxedSamplable<F, R> = Arc<dyn Samplable<F, R> + Send + Sync>;
pub type BoxedMaterial<F, R> = Arc<dyn Material<F, R> + Send + Sync>;
pub type BoxedTexture<F> = Arc<dyn Texture<F> + Send + Sync>;

pub trait SimdF32Field: SimdRealField<Element = f32> + MySimdVector {}

pub trait SimdBoolField<F: SimdValue>:
    SimdValue<Element = bool, SimdBool = F::SimdBool> + Debug
{
}

impl<T> SimdF32Field for T where T: SimdRealField<Element = f32> + MySimdVector {}

impl<T, F: SimdValue> SimdBoolField<F> for T where
    T: SimdValue<Element = bool, SimdBool = F::SimdBool> + Debug
{
}

#[macro_export]
macro_rules! extract {
    ($rays:ident, $mapper:expr) => {
        unsafe {
            $rays
                .iter()
                .map($mapper)
                .collect::<ArrayVec<_, { F::LANES }>>()
                .into_inner_unchecked()
        }
    };
}
