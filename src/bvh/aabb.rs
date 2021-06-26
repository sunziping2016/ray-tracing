use crate::py::PyVector3;
use nalgebra::{Point3, Vector3};
use pyo3::proc_macro::{pyclass, pymethods};

#[pyclass(name = "AABB")]
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

#[pymethods]
impl AABB {
    #[new]
    fn py_new(min: PyVector3, max: PyVector3) -> Self {
        Self::with_bounds(
            Point3::new(min.0, min.1, min.2),
            Point3::new(max.0, max.1, max.2),
        )
    }
    #[getter("min")]
    fn py_min(&self) -> PyVector3 {
        (self.min[0], self.min[1], self.min[2])
    }
    #[getter("max")]
    fn py_max(&self) -> PyVector3 {
        (self.max[0], self.max[1], self.max[2])
    }
}

impl AABB {
    pub fn with_bounds(min: Point3<f32>, max: Point3<f32>) -> Self {
        Self { min, max }
    }
    pub fn empty() -> Self {
        Self {
            min: Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.min[0] > self.max[0] || self.min[1] > self.max[1] || self.min[2] > self.max[2]
    }
    pub fn join(&self, other: &Self) -> Self {
        let min = self.min.inf(&other.min);
        let max = self.max.sup(&other.max);
        Self { min, max }
    }
    pub fn grow(&self, other: &Point3<f32>) -> Self {
        let min = self.min.inf(other);
        let max = self.max.sup(other);
        Self { min, max }
    }
    pub fn size(&self) -> Vector3<f32> {
        self.max - self.min
    }
    pub fn center(&self) -> Point3<f32> {
        self.min + self.size().scale(0.5)
    }
    pub fn surface_area(&self) -> f32 {
        2.0 * self.size().norm_squared()
    }
}
