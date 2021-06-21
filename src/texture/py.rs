use crate::py::PyBoxedTexture;
use crate::texture::solid_color::SolidColor;
use pyo3::types::PyModule;
use pyo3::{Py, PyAny, PyResult, Python};

pub fn py_init_texture(module: &PyModule) -> PyResult<()> {
    module.add_class::<SolidColor>()?;
    Ok(())
}

pub fn to_texture(py: Python, item: Py<PyAny>) -> PyBoxedTexture {
    if let Ok(solid_color) = item.extract::<Py<SolidColor>>(py) {
        Box::new(solid_color) as PyBoxedTexture
    } else {
        todo!()
    }
}
