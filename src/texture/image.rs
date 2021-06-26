use crate::texture::Texture;
use crate::SimdF32Field;
use image::{GenericImageView, Pixel};
use nalgebra::{Point3, SimdBool, Vector2, Vector3};
use num_traits::NumCast;
use std::array::IntoIter;

#[derive(Debug, Clone)]
pub struct ImageTexture<T> {
    image: T,
}

impl<T> ImageTexture<T> {
    pub fn new(image: T) -> Self {
        ImageTexture { image }
    }
}

impl<T, F> Texture<F> for ImageTexture<T>
where
    T: GenericImageView,
    F: SimdF32Field + Into<[f32; F::LANES]> + From<[f32; F::LANES]>,
{
    #[allow(clippy::many_single_char_names, clippy::uninit_assumed_init)]
    fn value(&self, uv: &Vector2<F>, _p: &Point3<F>) -> Vector3<F> {
        let u = uv[0].is_simd_negative().if_else(F::zero, || {
            uv[0].simd_gt(F::one()).if_else(F::one, || uv[0])
        });
        let v = F::one()
            - uv[1].is_simd_negative().if_else(F::zero, || {
                uv[1].simd_gt(F::one()).if_else(F::one, || uv[1])
            });
        let i = Into::<[f32; F::LANES]>::into(F::splat(self.image.width() as f32) * u).map(|x| {
            let x = x as u32;
            if x < self.image.width() {
                x
            } else {
                self.image.width() - 1
            }
        });
        let j = Into::<[f32; F::LANES]>::into(F::splat(self.image.height() as f32) * v).map(|x| {
            let x = x as u32;
            if x < self.image.height() {
                x
            } else {
                self.image.height() - 1
            }
        });
        let mut r: [f32; F::LANES] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let mut g: [f32; F::LANES] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let mut b: [f32; F::LANES] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        IntoIter::new(i)
            .zip(IntoIter::new(j))
            .enumerate()
            .for_each(|(idx, (i, j))| {
                let rgb = self.image.get_pixel(i, j).to_rgb();
                unsafe {
                    std::ptr::write(
                        &mut r[idx],
                        <f32 as NumCast>::from(rgb.0[0]).unwrap() / 255f32,
                    );
                    std::ptr::write(
                        &mut g[idx],
                        <f32 as NumCast>::from(rgb.0[1]).unwrap() / 255f32,
                    );
                    std::ptr::write(
                        &mut b[idx],
                        <f32 as NumCast>::from(rgb.0[2]).unwrap() / 255f32,
                    );
                }
            });
        Vector3::new(F::from(r), F::from(g), F::from(b))
    }
}
