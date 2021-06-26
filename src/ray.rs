use crate::bvh::aabb::AABB;
use crate::extract;
use crate::py::{bits_to_m, bits_to_numpy, PySimd};
use crate::simd::MySimdVector;
use arrayvec::ArrayVec;
use nalgebra::{
    ClosedAdd, Point3, Scalar, SimdBool, SimdRealField, SimdValue, UnitVector3, Vector3,
};
use num_traits::{One, Zero};
use numpy::{PyArray1, PyArray2};
use pyo3::proc_macro::{pyclass, pymethods};
use pyo3::Python;

#[derive(Debug, Clone)]
pub struct Ray<F: SimdValue> {
    origin: Point3<F>,
    direction: UnitVector3<F>,
    time: F,
    mask: F::SimdBool,
}

impl<F: SimdValue> From<&[Ray<f32>]> for Ray<F>
where
    F: MySimdVector + From<[f32; F::LANES]>,
    F::SimdBool: From<[bool; F::LANES]>,
{
    fn from(rays: &[Ray<f32>]) -> Self {
        let origin_x = extract!(rays, |x| x.origin[0]);
        let origin_y = extract!(rays, |x| x.origin[1]);
        let origin_z = extract!(rays, |x| x.origin[2]);
        let direction_x = extract!(rays, |x| x.direction[0]);
        let direction_y = extract!(rays, |x| x.direction[1]);
        let direction_z = extract!(rays, |x| x.direction[2]);
        let time = extract!(rays, |x| x.time);
        let mask = extract!(rays, |x| x.mask);
        Self {
            origin: Point3::new(F::from(origin_x), F::from(origin_y), F::from(origin_z)),
            direction: UnitVector3::new_unchecked(Vector3::new(
                F::from(direction_x),
                F::from(direction_y),
                F::from(direction_z),
            )),
            time: F::from(time),
            mask: F::SimdBool::from(mask),
        }
    }
}

impl<F: SimdValue> Default for Ray<F>
where
    F: Zero + Scalar + ClosedAdd,
    F::Element: Zero + One + Scalar,
    F::SimdBool: SimdValue<Element = bool>,
{
    fn default() -> Self {
        Self {
            origin: Point3::new(F::zero(), F::zero(), F::zero()),
            direction: UnitVector3::new_unchecked(Vector3::splat(Vector3::x())),
            time: Zero::zero(),
            mask: F::SimdBool::splat(false),
        }
    }
}

impl<F: SimdValue> SimdValue for Ray<F>
where
    F: Scalar,
    F::Element: Scalar,
    F::SimdBool: SimdValue<Element = bool, SimdBool = F::SimdBool>,
{
    type Element = Ray<F::Element>;
    type SimdBool = F::SimdBool;

    fn lanes() -> usize {
        F::lanes()
    }
    fn splat(val: Self::Element) -> Self {
        Self::new(
            Point3::splat(val.origin),
            UnitVector3::new_unchecked(Vector3::splat(val.direction.into_inner())),
            F::splat(val.time),
            F::SimdBool::splat(val.mask),
        )
    }
    fn extract(&self, i: usize) -> Self::Element {
        Ray::new(
            self.origin.extract(i),
            UnitVector3::new_unchecked(Vector3::splat(self.direction.as_ref().extract(i))),
            self.time.extract(i),
            self.mask.extract(i),
        )
    }
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        Ray::new(
            self.origin.extract_unchecked(i),
            UnitVector3::new_unchecked(Vector3::splat(
                self.direction.as_ref().extract_unchecked(i),
            )),
            self.time.extract_unchecked(i),
            self.mask.extract_unchecked(i),
        )
    }
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.origin.replace(i, val.origin);
        self.direction
            .as_mut_unchecked()
            .replace(i, val.direction.into_inner());
        self.time.replace(i, val.time);
        self.mask.replace(i, val.mask);
    }
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.origin.replace_unchecked(i, val.origin);
        self.direction
            .as_mut_unchecked()
            .replace_unchecked(i, val.direction.into_inner());
        self.time.replace_unchecked(i, val.time);
        self.mask.replace_unchecked(i, val.mask);
    }
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        Self {
            origin: self.origin.select(cond, other.origin),
            direction: UnitVector3::new_unchecked(
                self.direction
                    .into_inner()
                    .select(cond, other.direction.into_inner()),
            ),
            time: self.time.select(cond, other.time),
            mask: self.mask.select(cond, other.mask),
        }
    }
}

impl<F: SimdValue> Ray<F> {
    pub fn new(origin: Point3<F>, direction: UnitVector3<F>, time: F, mask: F::SimdBool) -> Self {
        Ray {
            origin,
            direction,
            time,
            mask,
        }
    }
    pub fn at(&self, t: F) -> Point3<F>
    where
        F: SimdRealField,
    {
        self.origin + self.direction.scale(t)
    }
    pub fn origin(&self) -> &Point3<F> {
        &self.origin
    }
    pub fn direction(&self) -> &UnitVector3<F> {
        &self.direction
    }
    pub fn time(&self) -> &F {
        &self.time
    }
    pub fn mask(&self) -> F::SimdBool {
        self.mask
    }
}

impl<F: SimdValue<Element = f32>> Ray<F> {
    pub fn intersects_aabb(&self, aabb: &AABB, mut t_min: F, mut t_max: F) -> F::SimdBool
    where
        F: SimdRealField,
    {
        let mut update_and_test = |min: f32, max: f32, origin: F, direction: F| -> F::SimdBool {
            let min_val = (F::splat(min) - origin) / direction;
            let max_val = (F::splat(max) - origin) / direction;
            t_min = min_val.simd_min(max_val).simd_max(t_min);
            t_max = min_val.simd_max(max_val).simd_min(t_max);
            t_min.simd_lt(t_max)
        };
        let mask = self.mask
            & update_and_test(aabb.min[0], aabb.max[0], self.origin[0], self.direction[0]);
        if mask.none() {
            return mask;
        }
        let mask =
            mask & update_and_test(aabb.min[1], aabb.max[1], self.origin[1], self.direction[1]);
        if mask.none() {
            return mask;
        }
        mask & update_and_test(aabb.min[2], aabb.max[2], self.origin[2], self.direction[2])
    }
}

#[pyclass(name = "Ray")]
#[derive(Debug, Clone)]
pub struct PyRay {
    origin: Vec<f32>,
    direction: Vec<f32>,
    time: [f32; PySimd::LANES],
    mask: u64,
}

impl From<&Ray<PySimd>> for PyRay {
    fn from(ray: &Ray<PySimd>) -> Self {
        let origin = [
            Into::<[f32; PySimd::LANES]>::into(ray.origin[0]),
            ray.origin[1].into(),
            ray.origin[2].into(),
        ]
        .concat();
        let direction = [
            Into::<[f32; PySimd::LANES]>::into(ray.direction[0]),
            ray.direction.as_ref()[1].into(),
            ray.direction.as_ref()[2].into(),
        ]
        .concat();
        let time: [f32; PySimd::LANES] = ray.time.into();
        let mask = ray.mask.bitmask();
        Self {
            origin,
            direction,
            time,
            mask,
        }
    }
}

impl From<&PyRay> for Ray<PySimd> {
    fn from(ray: &PyRay) -> Self {
        let origin = Point3::new(
            PySimd::from_slice_unaligned(&ray.origin[0..PySimd::LANES]),
            PySimd::from_slice_unaligned(&ray.origin[PySimd::LANES..(PySimd::LANES * 2)]),
            PySimd::from_slice_unaligned(&ray.origin[(PySimd::LANES * 2)..(PySimd::LANES * 3)]),
        );
        let direction = UnitVector3::new_normalize(Vector3::new(
            PySimd::from_slice_unaligned(&ray.direction[0..PySimd::LANES]),
            PySimd::from_slice_unaligned(&ray.direction[PySimd::LANES..(PySimd::LANES * 2)]),
            PySimd::from_slice_unaligned(&ray.direction[(PySimd::LANES * 2)..(PySimd::LANES * 3)]),
        ));
        let time = PySimd::from(ray.time);
        let mask = bits_to_m::<PySimd>(ray.mask);
        Self {
            origin,
            direction,
            time,
            mask,
        }
    }
}

#[pymethods]
impl PyRay {
    #[getter("origin")]
    fn py_origin<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        PyArray1::from_slice(py, &self.origin)
            .reshape([3, PySimd::LANES])
            .unwrap()
    }
    #[getter("direction")]
    fn py_direction<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        PyArray1::from_slice(py, &self.direction)
            .reshape([3, PySimd::LANES])
            .unwrap()
    }
    #[getter("time")]
    fn py_time<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        PyArray1::from_slice(py, &self.time)
    }
    #[getter("mask")]
    fn py_mask<'py>(&self, py: Python<'py>) -> &'py PyArray1<bool> {
        bits_to_numpy::<PySimd>(py, self.mask)
    }
}
