#![allow(incomplete_features)]
#![feature(never_type)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]

pub mod bvh;
pub mod camera;
pub mod image;
pub mod ray;
pub mod simd;

pub const EPSILON: f32 = 0.00001;
