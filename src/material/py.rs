use crate::material::dielectric::Dielectric;
use crate::material::lambertian::PyLambertian;
use crate::material::metal::Metal;
use crate::py::PyBoxedMaterial;
use pyo3::types::PyModule;
use pyo3::{Py, PyAny, PyResult, Python};
use std::sync::Arc;

pub fn py_init_material(module: &PyModule) -> PyResult<()> {
    module.add_class::<PyLambertian>()?;
    module.add_class::<Dielectric>()?;
    module.add_class::<Metal>()?;
    Ok(())
}

pub fn to_material(py: Python, item: Py<PyAny>) -> PyBoxedMaterial {
    if let Ok(lambertian) = item.extract::<Py<PyLambertian>>(py) {
        Arc::new(lambertian) as PyBoxedMaterial
    } else if let Ok(dielectric) = item.extract::<Py<Dielectric>>(py) {
        Arc::new(dielectric) as PyBoxedMaterial
    } else if let Ok(metal) = item.extract::<Py<Metal>>(py) {
        Arc::new(metal) as PyBoxedMaterial
    } else {
        todo!()
    }
}
