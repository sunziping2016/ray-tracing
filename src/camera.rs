use crate::py::PyVector3;
use crate::random::{random_in_unit_disk, random_uniform};
use crate::ray::Ray;
use crate::simd::MySimdVector;
use nalgebra::{Point3, SimdValue};
use nalgebra::{SimdRealField, Unit, Vector2, Vector3};
use pyo3::proc_macro::{pyclass, pymethods};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f32::consts;

#[pyclass(name = "PerspectiveCameraParam")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraParam {
    pub look_from: Point3<f32>,
    pub look_at: Point3<f32>,
    pub vfov: f32,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub up: Option<Vector3<f32>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub aspect_ratio: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub aperture: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub focus_dist: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub time0: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub time1: Option<f32>,
}

#[pymethods]
impl CameraParam {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        look_from: PyVector3,
        look_at: PyVector3,
        vfov: f32,
        up: Option<PyVector3>,
        aspect_ratio: Option<f32>,
        aperture: Option<f32>,
        focus_dist: Option<f32>,
        time0: Option<f32>,
        time1: Option<f32>,
    ) -> Self {
        Self {
            look_from: Point3::new(look_from.0, look_from.1, look_from.2),
            look_at: Point3::new(look_at.0, look_at.1, look_at.2),
            vfov,
            up: up.map(|x| Vector3::new(x.0, x.1, x.2)),
            aspect_ratio,
            aperture,
            focus_dist,
            time0,
            time1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Camera {
    origin: Point3<f32>,
    lower_left_corner: Point3<f32>,
    horizontal: Vector3<f32>,
    vertical: Vector3<f32>,
    u: Vector3<f32>, // norm
    v: Vector3<f32>, // norm
    lens_radius: f32,
    time0: f32,
    time1: f32,
}

impl Camera {
    pub fn new(param: CameraParam, default_aspect_ratio: f32) -> Self {
        let theta = param.vfov * consts::PI / 180.0;
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let aspect_ratio = param.aspect_ratio.unwrap_or(default_aspect_ratio);
        let viewport_width = aspect_ratio * viewport_height;

        let w = (param.look_from - param.look_at).normalize();
        let up = param.up.unwrap_or_else(|| Vector3::new(0.0, 1.0, 0.0));
        let u = up.cross(&w).normalize();
        let v = w.cross(&u).normalize();

        let focus_dist = param
            .focus_dist
            .unwrap_or_else(|| (param.look_from - param.look_at).norm());
        let horizontal = u.scale(focus_dist * viewport_width);
        let vertical = v.scale(focus_dist * viewport_height);
        let lower_left_corner =
            param.look_from - horizontal.unscale(2.0) - vertical.unscale(2.0) - w.scale(focus_dist);

        let aperture = param.aperture.unwrap_or(0.0);
        let lens_radius = aperture / 2.0;

        Self {
            origin: param.look_from,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            lens_radius,
            time0: param.time0.unwrap_or(0.0),
            time1: param.time1.unwrap_or(0.0),
        }
    }
    pub fn get_ray<F, R: Rng>(&self, st: Vector2<F>, mask: F::SimdBool, rng: &mut R) -> Ray<F>
    where
        F: SimdRealField<Element = f32> + MySimdVector,
    {
        let rd = random_in_unit_disk::<F, _>(rng).scale(F::splat(self.lens_radius));
        let offset = Vector3::splat(self.u).scale(rd[0]) + Vector3::splat(self.v).scale(rd[1]);
        let source = Point3::splat(self.origin) + offset;
        let target = Point3::splat(self.lower_left_corner)
            + Vector3::splat(self.horizontal).scale(st[0])
            + Vector3::splat(self.vertical).scale(st[1]);
        Ray::new(
            source,
            Unit::new_normalize(target - source),
            random_uniform::<F, _, _>(self.time0..=self.time1, rng),
            mask,
        )
    }
}
