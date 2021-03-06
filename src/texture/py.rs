use crate::py::PyBoxedTexture;
use crate::texture::checker::PyChecker;
use crate::texture::solid_color::SolidColor;
use pyo3::types::PyModule;
use pyo3::{Py, PyAny, PyResult, Python};
use std::sync::Arc;

pub fn py_init_texture(module: &PyModule) -> PyResult<()> {
    module.add_class::<SolidColor>()?;
    module.add_class::<PyChecker>()?;
    Ok(())
}

pub fn to_texture(py: Python, item: Py<PyAny>) -> PyBoxedTexture {
    if let Ok(solid_color) = item.extract::<Py<SolidColor>>(py) {
        Arc::new(
            unsafe { solid_color.into_ref(py).try_borrow_unguarded() }
                .unwrap()
                .clone(),
        ) as PyBoxedTexture
    } else if let Ok(checker) = item.extract::<Py<PyChecker>>(py) {
        Arc::new(
            unsafe { checker.into_ref(py).try_borrow_unguarded() }
                .unwrap()
                .clone(),
        ) as PyBoxedTexture
    } else {
        todo!()
    }
}
