use crate::simd_f32::SimdF32Field;
use itertools::iproduct;
use nalgebra::Vector2;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::iter;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageParam {
    width: usize,
    height: usize,
}

impl ImageParam {
    pub fn new(width: usize, height: usize) -> Self {
        ImageParam { width, height }
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
    pub fn sample<S: SimdF32Field, R: Rng>(&self, rng: &mut R) -> Vec<(Vector2<S>, S::SimdBool)> {
        let width = self.width as f32;
        let height = self.height as f32;
        let (ys, xs): (Vec<_>, Vec<_>) = iproduct!(0..self.height, 0..self.width)
            .map(|(i, j)| {
                (
                    rng.gen_range(((i as f32 - 0.5) / width)..((i as f32 + 0.5) / width)),
                    rng.gen_range(((j as f32 - 0.5) / height)..((j as f32 + 0.5) / height)),
                )
            })
            .chain(iter::repeat((f32::NAN, f32::NAN)).take(S::lanes() - 1))
            .unzip();
        xs.chunks_exact(S::lanes())
            .map(|chunk| S::from_f32_slice(chunk))
            .zip(
                ys.chunks_exact(S::lanes())
                    .map(|chunk| S::from_f32_slice(chunk)),
            )
            .map(|(x, y)| (Vector2::new(x, y), !x.is_nan()))
            .collect()
    }
}
