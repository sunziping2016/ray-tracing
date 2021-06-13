use nalgebra::{Unit, UnitVector3, Vector3};
use num_traits::float::FloatConst;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraParam {
    look_from: Vector3<f32>,
    look_at: Vector3<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    up: Option<Vector3<f32>>,
    vfov: f32,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    aspect_ratio: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    aperture: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    focus_dist: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    time0: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    time1: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct Camera {
    origin: Vector3<f32>,
    lower_left_corner: Vector3<f32>,
    horizontal: Vector3<f32>,
    vertical: Vector3<f32>,
    u: UnitVector3<f32>,
    v: UnitVector3<f32>,
    w: UnitVector3<f32>,
    lens_radius: f32,
    time0: f32,
    time1: f32,
}

impl Camera {
    pub fn new(param: CameraParam, default_aspect_ratio: f32) -> Self {
        let theta = param.vfov * f32::PI() / 180.0;
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let aspect_ratio = param.aspect_ratio.unwrap_or(default_aspect_ratio);
        let viewport_width = aspect_ratio * viewport_height;

        let w = Unit::new_normalize(param.look_from - param.look_at);
        let up = param.up.unwrap_or_else(|| Vector3::new(0.0, 1.0, 0.0));
        let u = Unit::new_normalize(up.cross(&w));
        let v = Unit::new_normalize(w.cross(&u));

        let focus_dist = param
            .focus_dist
            .unwrap_or_else(|| (param.look_from - param.look_at).norm());
        let horizontal = u.scale(focus_dist * viewport_height);
        let vertical = v.scale(focus_dist * viewport_width);
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
            w,
            lens_radius,
            time0: param.time0.unwrap_or(0.0),
            time1: param.time1.unwrap_or(0.0),
        }
    }
}
