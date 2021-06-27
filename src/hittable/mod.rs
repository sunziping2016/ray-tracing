pub mod aa_rect;
pub mod constant_medium;
#[cfg(feature = "python")]
pub mod py;
pub mod sphere;
pub mod transform;
pub mod triangle;

use crate::bvh::aabb::AABB;
use crate::extract;
use crate::ray::Ray;
use crate::simd::MySimdVector;
use crate::SimdF32Field;
use arrayvec::ArrayVec;
use auto_impl::auto_impl;
use nalgebra::{
    ClosedAdd, Point3, Scalar, SimdBool, SimdRealField, SimdValue, UnitVector3, Vector2, Vector3,
};
use num_traits::Zero;
#[cfg(feature = "python")]
use pyo3::{Py, PyClass, Python};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct HitRecord<F: SimdValue> {
    pub p: Point3<F>,
    pub normal: UnitVector3<F>,
    pub t: F,
    pub uv: Vector2<F>,
    pub front_face: F::SimdBool,
    pub mask: F::SimdBool,
}

impl<F: SimdValue> From<&[HitRecord<f32>]> for HitRecord<F>
where
    F: MySimdVector + From<[f32; F::LANES]>,
    F::SimdBool: From<[bool; F::LANES]>,
{
    fn from(records: &[HitRecord<f32>]) -> Self {
        let p_x = extract!(records, |x| x.p[0]);
        let p_y = extract!(records, |x| x.p[1]);
        let p_z = extract!(records, |x| x.p[2]);
        let normal_x = extract!(records, |x| x.normal[0]);
        let normal_y = extract!(records, |x| x.normal[1]);
        let normal_z = extract!(records, |x| x.normal[2]);
        let t = extract!(records, |x| x.t);
        let u = extract!(records, |x| x.uv[0]);
        let v = extract!(records, |x| x.uv[1]);
        let front_face = extract!(records, |x| x.front_face);
        let mask = extract!(records, |x| x.mask);
        Self {
            p: Point3::new(F::from(p_x), F::from(p_y), F::from(p_z)),
            normal: UnitVector3::new_unchecked(Vector3::new(
                F::from(normal_x),
                F::from(normal_y),
                F::from(normal_z),
            )),
            t: F::from(t),
            uv: Vector2::new(F::from(u), F::from(v)),
            front_face: F::SimdBool::from(front_face),
            mask: F::SimdBool::from(mask),
        }
    }
}

impl<F: SimdValue> SimdValue for HitRecord<F>
where
    F: Scalar,
    F::Element: Scalar,
    F::SimdBool: SimdValue<Element = bool, SimdBool = F::SimdBool>,
{
    type Element = HitRecord<F::Element>;
    type SimdBool = F::SimdBool;

    fn lanes() -> usize {
        F::lanes()
    }
    fn splat(val: Self::Element) -> Self {
        Self {
            p: Point3::splat(val.p),
            normal: UnitVector3::new_unchecked(Vector3::splat(val.normal.into_inner())),
            t: F::splat(val.t),
            uv: Vector2::splat(val.uv),
            front_face: F::SimdBool::splat(val.front_face),
            mask: F::SimdBool::splat(val.mask),
        }
    }
    fn extract(&self, i: usize) -> Self::Element {
        HitRecord {
            p: self.p.extract(i),
            normal: UnitVector3::new_unchecked(self.normal.as_ref().extract(i)),
            t: self.t.extract(i),
            uv: self.uv.extract(i),
            front_face: self.front_face.extract(i),
            mask: self.mask.extract(i),
        }
    }
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        HitRecord {
            p: self.p.extract_unchecked(i),
            normal: UnitVector3::new_unchecked(self.normal.as_ref().extract_unchecked(i)),
            t: self.t.extract_unchecked(i),
            uv: self.uv.extract_unchecked(i),
            front_face: self.front_face.extract_unchecked(i),
            mask: self.mask.extract_unchecked(i),
        }
    }
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.p.replace(i, val.p);
        self.normal
            .as_mut_unchecked()
            .replace(i, val.normal.into_inner());
        self.t.replace(i, val.t);
        self.uv.replace(i, val.uv);
        self.front_face.replace(i, val.front_face);
        self.mask.replace(i, val.mask);
    }
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.p.replace_unchecked(i, val.p);
        self.normal
            .as_mut_unchecked()
            .replace_unchecked(i, val.normal.into_inner());
        self.t.replace_unchecked(i, val.t);
        self.uv.replace_unchecked(i, val.uv);
        self.front_face.replace_unchecked(i, val.front_face);
        self.mask.replace_unchecked(i, val.mask);
    }

    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        Self {
            p: self.p.select(cond, other.p),
            normal: UnitVector3::new_unchecked(
                self.normal
                    .into_inner()
                    .select(cond, other.normal.into_inner()),
            ),
            t: self.t.select(cond, other.t),
            uv: self.uv.select(cond, other.uv),
            front_face: self.front_face.select(cond, other.front_face),
            mask: self.mask.select(cond, other.mask),
        }
    }
}

impl<F: SimdF32Field> HitRecord<F> {
    pub fn face_normal(
        direction: &UnitVector3<F>,
        outward_normal: UnitVector3<F>,
    ) -> (F::SimdBool, UnitVector3<F>) {
        let front_face = direction.dot(&outward_normal).is_simd_negative();
        let outward_normal = outward_normal.into_inner();
        let normal = front_face.if_else(|| outward_normal, || -outward_normal);
        (front_face, UnitVector3::new_unchecked(normal))
    }
}

impl<F: SimdValue> Default for HitRecord<F>
where
    F: Zero + Scalar + ClosedAdd + SimdValue<Element = f32>,
    F::SimdBool: SimdValue<Element = bool>,
{
    fn default() -> Self {
        Self {
            p: Point3::new(F::zero(), F::zero(), F::zero()),
            normal: UnitVector3::new_unchecked(Vector3::splat(Vector3::y())),
            t: F::splat(f32::INFINITY),
            uv: Zero::zero(),
            front_face: F::SimdBool::splat(false),
            mask: F::SimdBool::splat(false),
        }
    }
}

#[auto_impl(&, &mut, Box, Rc, Arc)]
pub trait Bounded {
    fn bounding_box(&self, time0: f32, time1: f32) -> AABB;
}

#[auto_impl(&, &mut, Box, Rc, Arc)]
pub trait Hittable<F: SimdValue, R: Rng>: Bounded {
    fn hit(&self, ray: &Ray<F>, t_min: F, t_max: F, rng: &mut R) -> HitRecord<F>;
}

#[auto_impl(&, &mut, Box, Rc, Arc)]
pub trait Samplable<F: SimdRealField, R: Rng> {
    fn value(
        &self,
        _origin: &Point3<F>,
        _direction: &UnitVector3<F>,
        _mask: F::SimdBool,
        _rng: &mut R,
    ) -> F {
        F::zero()
    }
    fn generate(&self, _origin: &Point3<F>, _rng: &mut R) -> UnitVector3<F> {
        UnitVector3::new_unchecked(Vector3::new(F::one(), F::zero(), F::zero()))
    }
}

#[cfg(feature = "python")]
impl<T> Bounded for Py<T>
where
    T: Bounded + PyClass,
{
    fn bounding_box(&self, time0: f32, time1: f32) -> AABB {
        Python::with_gil(|py| self.as_ref(py).borrow().bounding_box(time0, time1))
    }
}

#[cfg(feature = "python")]
impl<T, F: SimdRealField, R: Rng> Hittable<F, R> for Py<T>
where
    T: Hittable<F, R> + PyClass,
{
    fn hit(&self, ray: &Ray<F>, t_min: F, t_max: F, rng: &mut R) -> HitRecord<F> {
        Python::with_gil(|py| self.borrow(py).hit(ray, t_min, t_max, rng))
    }
}

#[cfg(feature = "python")]
impl<T, F: SimdRealField, R: Rng> Samplable<F, R> for Py<T>
where
    T: Samplable<F, R> + PyClass,
{
    fn value(
        &self,
        origin: &Point3<F>,
        direction: &UnitVector3<F>,
        mask: F::SimdBool,
        rng: &mut R,
    ) -> F {
        Python::with_gil(|py| self.borrow(py).value(origin, direction, mask, rng))
    }

    fn generate(&self, origin: &Point3<F>, rng: &mut R) -> UnitVector3<F> {
        Python::with_gil(|py| self.borrow(py).generate(origin, rng))
    }
}
