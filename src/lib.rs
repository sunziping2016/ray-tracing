#![allow(incomplete_features)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]

pub mod bvh;
pub mod camera;
pub mod image;
pub mod random;
pub mod ray;
pub mod simd;

pub const EPSILON: f32 = 0.00001;
