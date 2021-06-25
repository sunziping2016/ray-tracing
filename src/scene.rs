use crate::bvh::bvh::BVH;
use crate::hittable::py::to_hittable;
use crate::material::py::to_material;
use crate::py::{PyRng, PySimd, PyVector3};
use crate::{BoxedHittable, BoxedMaterial};
use nalgebra::Vector3;
use num_traits::Zero;
use pyo3::proc_macro::{pyclass, pymethods};
use pyo3::{Py, PyAny, Python};
use rand::Rng;

#[derive(Clone)]
pub struct Scene<F, R: Rng> {
    hittables: Vec<BoxedHittable<F, R>>,
    materials: Vec<BoxedMaterial<F, R>>,
    background: Vector3<f32>,
    environment: Vector3<f32>,
}

impl<F, R: Rng> Scene<F, R> {
    pub fn new(background: Vector3<f32>, environment: Vector3<f32>) -> Self {
        Self {
            hittables: Vec::new(),
            materials: Vec::new(),
            background,
            environment,
        }
    }
    pub fn add(&mut self, hittable: BoxedHittable<F, R>, material: BoxedMaterial<F, R>) {
        self.hittables.push(hittable);
        self.materials.push(material);
    }
    pub fn build_bvh(&self, time0: f32, time1: f32) -> BVH {
        BVH::build(
            &self
                .hittables
                .iter()
                .map(|x| x.bounding_box(time0, time1))
                .collect::<Vec<_>>(),
        )
    }
    pub fn len(&self) -> usize {
        self.hittables.len()
    }
    pub fn is_empty(&self) -> bool {
        self.hittables.is_empty()
    }
    pub fn hittable(&self, index: usize) -> &BoxedHittable<F, R> {
        &self.hittables[index]
    }
    pub fn material(&self, index: usize) -> &BoxedMaterial<F, R> {
        &self.materials[index]
    }
    pub fn background(&self) -> Vector3<f32> {
        self.background
    }
    pub fn environment(&self) -> Vector3<f32> {
        self.environment
    }
}

#[pyclass(name = "Scene")]
pub struct PyScene {
    pub inner: Scene<PySimd, PyRng>,
}

#[pymethods]
impl PyScene {
    // TODO: iterable
    #[new]
    pub fn py_new(background: PyVector3, environment: Option<PyVector3>) -> Self {
        Self {
            inner: Scene::new(
                Vector3::new(background.0, background.1, background.2),
                environment.map_or(Zero::zero(), |(r, g, b)| Vector3::new(r, g, b)),
            ),
        }
    }
    #[name = "add"]
    pub fn py_add(&mut self, py: Python, shape: Py<PyAny>, material: Py<PyAny>) {
        let hittable = to_hittable(py, shape);
        let material = to_material(py, material);
        self.inner.add(hittable, material);
    }
}
