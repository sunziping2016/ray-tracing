use crate::material::lambertian::PyLambertian;
use crate::py::PyBoxedMaterial;
use pyo3::types::PyModule;
use pyo3::{Py, PyAny, PyResult, Python};
use std::sync::Arc;

pub fn py_init_material(module: &PyModule) -> PyResult<()> {
    module.add_class::<PyLambertian>()?;
    Ok(())
}

pub fn to_material(py: Python, item: Py<PyAny>) -> PyBoxedMaterial {
    if let Ok(lambertian) = item.extract::<Py<PyLambertian>>(py) {
        Arc::new(lambertian) as PyBoxedMaterial
    } else {
        todo!()
    }
}
