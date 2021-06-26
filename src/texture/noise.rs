use crate::random::random_on_unit_sphere;
use crate::simd::MySimdVector;
use crate::texture::Texture;
use arrayvec::ArrayVec;
use itertools::{iproduct, izip};
use nalgebra::{Point3, SimdRealField, Vector2, Vector3};
use rand::Rng;
use std::array::IntoIter;

pub const POINT_COUNT: usize = 256;

#[derive(Debug, Clone)]
pub struct Perlin {
    rand_vec: Vec<Vector3<f32>>,
    perm_x: Vec<usize>,
    perm_y: Vec<usize>,
    perm_z: Vec<usize>,
}

impl Perlin {
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        let rand_vec = (0..POINT_COUNT)
            .map(|_| random_on_unit_sphere(rng))
            .collect();
        Self {
            rand_vec,
            perm_x: Perlin::perlin_generate_perm(rng),
            perm_y: Perlin::perlin_generate_perm(rng),
            perm_z: Perlin::perlin_generate_perm(rng),
        }
    }
    fn perlin_generate_perm<R: Rng>(rng: &mut R) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..POINT_COUNT).collect();
        (1..POINT_COUNT).rev().for_each(|i| {
            let target = rng.gen_range(0..=i);
            perm.swap(target, i);
        });
        perm
    }
    #[allow(clippy::many_single_char_names, clippy::uninit_assumed_init)]
    pub fn noise<F>(&self, p: &Vector3<F>) -> F
    where
        F: SimdRealField<Element = f32>
            + MySimdVector
            + Into<[f32; F::LANES]>
            + From<[f32; F::LANES]>,
    {
        macro_rules! to_usize_array {
            ($x:expr) => {
                unsafe {
                    IntoIter::new(Into::<[f32; F::LANES]>::into($x))
                        .map(|x| x as isize)
                        .collect::<ArrayVec<_, { F::LANES }>>()
                        .into_inner_unchecked()
                }
            };
        }
        let uvw = p - p.map(F::simd_floor);
        let i = to_usize_array!(p.map(F::simd_floor)[0]);
        let j = to_usize_array!(p.map(F::simd_floor)[1]);
        let k = to_usize_array!(p.map(F::simd_floor)[2]);
        let point_count = POINT_COUNT as isize;
        let array = iproduct!(0..2isize, 0..2isize, 0..2isize)
            .map(|(di, dj, dk)| {
                let mut x: [f32; F::LANES] =
                    unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                let mut y: [f32; F::LANES] =
                    unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                let mut z: [f32; F::LANES] =
                    unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                izip!(i.iter(), j.iter(), k.iter())
                    .enumerate()
                    .for_each(|(idx, (i, j, k))| {
                        let v = self.rand_vec[self.perm_x
                            [(i + di).rem_euclid(point_count) as usize]
                            ^ self.perm_y[(j + dj).rem_euclid(point_count) as usize]
                            ^ self.perm_z[(k + dk).rem_euclid(point_count) as usize]];
                        unsafe {
                            std::ptr::write(&mut x[idx], v[0]);
                            std::ptr::write(&mut y[idx], v[1]);
                            std::ptr::write(&mut z[idx], v[2]);
                        }
                    });
                Vector3::new(F::from(x), F::from(y), F::from(z))
            })
            .collect::<ArrayVec<_, 8>>();
        let c: [Vector3<F>; 8] = unsafe { array.into_inner_unchecked() };
        Perlin::perlin_interp(&c, &uvw)
    }

    pub fn turb<F>(&self, p: &Vector3<F>, depth: u32) -> F
    where
        F: SimdRealField<Element = f32>
            + MySimdVector
            + Into<[f32; F::LANES]>
            + From<[f32; F::LANES]>,
    {
        let mut accum = F::zero();
        let mut temp_p = *p;
        let mut weight = F::one();
        (0..depth).for_each(|_| {
            accum += weight * self.noise::<F>(&temp_p);
            weight *= F::splat(0.5f32);
            temp_p = temp_p.scale(F::splat(2f32));
        });
        accum.simd_abs()
    }

    fn perlin_interp<F>(c: &[Vector3<F>; 8], uvw: &Vector3<F>) -> F
    where
        F: SimdRealField<Element = f32>,
    {
        let uuvvww = uvw
            .component_mul(uvw)
            .component_mul(&(Vector3::from_element(F::splat(3f32)) - uvw.scale(F::splat(2f32))));
        let mut accum = F::zero();
        iproduct!(0..2usize, 0..2usize, 0..2usize)
            .enumerate()
            .for_each(|(idx, (i, j, k))| {
                let weight_v = Vector3::new(
                    uvw[0] - F::splat(i as f32),
                    uvw[1] - F::splat(j as f32),
                    uvw[2] - F::splat(k as f32),
                );
                accum += if i == 1 {
                    uuvvww[0]
                } else {
                    F::one() - uuvvww[0]
                } * if j == 1 {
                    uuvvww[1]
                } else {
                    F::one() - uuvvww[1]
                } * if k == 1 {
                    uuvvww[2]
                } else {
                    F::one() - uuvvww[2]
                } * c[idx].dot(&weight_v);
            });
        accum
    }
}

#[derive(Debug, Clone)]
pub struct Noise {
    noise: Perlin,
    scale: f32,
    depth: u32,
}

impl Noise {
    pub fn new(noise: Perlin, scale: f32, depth: u32) -> Self {
        Noise {
            noise,
            scale,
            depth,
        }
    }
}

impl<F> Texture<F> for Noise
where
    F: SimdRealField<Element = f32> + MySimdVector + Into<[f32; F::LANES]> + From<[f32; F::LANES]>,
{
    // TODO: add fn
    fn value(&self, _uv: &Vector2<F>, p: &Point3<F>) -> Vector3<F> {
        Vector3::from_element(F::one()).scale(
            self.noise
                .turb(&p.coords.scale(F::splat(self.scale)), self.depth),
        )
    }
}
