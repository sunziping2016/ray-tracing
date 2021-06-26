use crate::bvh::aabb::AABB;
use crate::hittable::sphere::Sphere;
use crate::hittable::{Bounded, HitRecord, Hittable};
use crate::py::{bits_to_m, f_to_numpy, PyBoxedHittable, PyRng, PySimd};
use crate::ray::{PyRay, Ray};
use crate::simd::MySimdVector;
use nalgebra::{Point3, SimdBool, SimdValue, UnitVector3, Vector2, Vector3};
use pyo3::proc_macro::pyclass;
use pyo3::types::PyModule;
use pyo3::{Py, PyAny, PyObject, PyResult, Python};
use std::sync::Arc;

pub struct PyHittable {
    inner: PyObject,
}

impl PyHittable {
    pub fn new(inner: PyObject) -> Self {
        Self { inner }
    }
}

impl Bounded for PyHittable {
    fn bounding_box(&self, time0: f32, time1: f32) -> AABB {
        Python::with_gil(|py| {
            let inner = self.inner.as_ref(py);
            inner
                .call_method1("bounding_box", (time0, time1))
                .unwrap()
                .extract()
        })
        .unwrap()
    }
}

impl Hittable<PySimd, PyRng> for PyHittable {
    fn hit(&self, ray: &Ray<PySimd>, t_min: PySimd, t_max: PySimd) -> HitRecord<PySimd> {
        Python::with_gil(|py| {
            let inner = self.inner.as_ref(py);
            (&inner
                .call_method1(
                    "hit",
                    (
                        PyRay::from(ray),
                        f_to_numpy(py, &t_min),
                        f_to_numpy(py, &t_max),
                    ),
                )
                .unwrap()
                .extract::<PyHitRecord>()
                .unwrap())
                .into()
        })
    }

    fn pdf_value(
        &self,
        _origin: &Point3<PySimd>,
        _direction: &UnitVector3<PySimd>,
        _mask: <PySimd as SimdValue>::SimdBool,
    ) -> PySimd {
        todo!()
    }

    fn random(&self, _rng: &mut PyRng, _origin: &Point3<PySimd>) -> Vector3<PySimd> {
        todo!()
    }
}

#[pyclass(name = "HitRecord")]
#[derive(Debug, Clone)]
pub struct PyHitRecord {
    pub p: Vec<f32>,
    pub normal: Vec<f32>,
    pub t: [f32; PySimd::LANES],
    pub uv: Vec<f32>,
    pub front_face: u64,
    pub mask: u64,
}

impl From<&HitRecord<PySimd>> for PyHitRecord {
    fn from(hit: &HitRecord<PySimd>) -> Self {
        let p = [
            Into::<[f32; PySimd::LANES]>::into(hit.p[0]),
            hit.p[1].into(),
            hit.p[2].into(),
        ]
        .concat();
        let normal = [
            Into::<[f32; PySimd::LANES]>::into(hit.normal[0]),
            hit.normal[1].into(),
            hit.normal[2].into(),
        ]
        .concat();
        let t: [f32; PySimd::LANES] = hit.t.into();
        let uv = [
            Into::<[f32; PySimd::LANES]>::into(hit.uv[0]),
            hit.uv[1].into(),
        ]
        .concat();
        let front_face = hit.mask.bitmask();
        let mask = hit.mask.bitmask();
        Self {
            p,
            normal,
            t,
            uv,
            front_face,
            mask,
        }
    }
}

impl From<&PyHitRecord> for HitRecord<PySimd> {
    fn from(hit: &PyHitRecord) -> Self {
        let p = Point3::new(
            PySimd::from_slice_unaligned(&hit.p[0..PySimd::LANES]),
            PySimd::from_slice_unaligned(&hit.p[PySimd::LANES..(PySimd::LANES * 2)]),
            PySimd::from_slice_unaligned(&hit.p[(PySimd::LANES * 2)..(PySimd::LANES * 3)]),
        );
        let normal = UnitVector3::new_normalize(Vector3::new(
            PySimd::from_slice_unaligned(&hit.normal[0..PySimd::LANES]),
            PySimd::from_slice_unaligned(&hit.normal[PySimd::LANES..(PySimd::LANES * 2)]),
            PySimd::from_slice_unaligned(&hit.normal[(PySimd::LANES * 2)..(PySimd::LANES * 3)]),
        ));
        let t = PySimd::from(hit.t);
        let uv = Vector2::new(
            PySimd::from_slice_unaligned(&hit.uv[0..PySimd::LANES]),
            PySimd::from_slice_unaligned(&hit.uv[PySimd::LANES..(PySimd::LANES * 2)]),
        );
        let front_face = bits_to_m::<PySimd>(hit.front_face);
        let mask = bits_to_m::<PySimd>(hit.mask);
        Self {
            p,
            normal,
            t,
            uv,
            front_face,
            mask,
        }
    }
}

pub fn py_init_hittable(module: &PyModule) -> PyResult<()> {
    module.add_class::<Sphere>()?;
    Ok(())
}

pub fn to_hittable(py: Python, item: Py<PyAny>) -> PyBoxedHittable {
    if let Ok(sphere) = item.extract::<Py<Sphere>>(py) {
        Arc::new(
            unsafe { sphere.into_ref(py).try_borrow_unguarded() }
                .unwrap()
                .clone(),
        ) as PyBoxedHittable
    } else {
        // Arc::new(PyHittable::new(item))
        todo!()
    }
}
