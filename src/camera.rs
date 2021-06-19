use crate::random::{random_in_unit_disk, random_uniform};
use crate::ray::Ray;
use crate::simd::MySimdVector;
use nalgebra::{SimdRealField, Unit, UnitVector3, Vector2, Vector3};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f32::consts;

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
pub struct Camera<F> {
    origin: Vector3<F>,
    lower_left_corner: Vector3<F>,
    horizontal: Vector3<F>,
    vertical: Vector3<F>,
    u: UnitVector3<F>,
    v: UnitVector3<F>,
    lens_radius: F,
    time0: f32,
    time1: f32,
}

impl<F> Camera<F> {
    pub fn new(param: CameraParam, default_aspect_ratio: f32) -> Self
    where
        F: SimdRealField<Element = f32>,
    {
        let theta = param.vfov * consts::PI / 180.0;
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let aspect_ratio = param.aspect_ratio.unwrap_or(default_aspect_ratio);
        let viewport_width = aspect_ratio * viewport_height;

        let w = Unit::new_normalize(param.look_from - param.look_at);
        let up = param.up.unwrap_or_else(|| Vector3::new(0.0, 1.0, 0.0));
        let u = Unit::new_normalize(up.cross(&w));
        let v: Unit<Vector3<f32>> = Unit::new_normalize(w.cross(&u));

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
            origin: param.look_from.map(F::splat),
            lower_left_corner: lower_left_corner.map(F::splat),
            horizontal: horizontal.map(F::splat),
            vertical: vertical.map(F::splat),
            u: Unit::new_unchecked(u.map(F::splat)),
            v: Unit::new_unchecked(v.map(F::splat)),
            lens_radius: F::splat(lens_radius),
            time0: param.time0.unwrap_or(0.0),
            time1: param.time1.unwrap_or(0.0),
        }
    }
    pub fn get_ray<R: Rng>(&self, st: Vector2<F>, mask: F::SimdBool, rng: &mut R) -> Ray<F>
    where
        F: SimdRealField<Element = f32> + MySimdVector,
    {
        let rd = random_in_unit_disk::<F, _>(rng).scale(self.lens_radius);
        let offset = self.u.scale(rd[0]) + self.v.scale(rd[1]);
        let source = self.origin + offset;
        let target =
            self.lower_left_corner + self.horizontal.scale(st[0]) + self.vertical.scale(st[1]);
        Ray::new(
            source,
            Unit::new_normalize(target - source),
            random_uniform::<F, _, _>(self.time0..=self.time1, rng),
            mask,
        )
    }
}
