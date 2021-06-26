use crate::bvh::aabb::AABB;
use crate::hittable::py::PyHitRecord;
use crate::hittable::{Bounded, HitRecord, Hittable};
use crate::py::{numpy_to_f, PyRng, PySimd, PyVector3};
use crate::random::random_to_sphere;
use crate::ray::{PyRay, Ray};
use crate::{SimdBoolField, SimdF32Field};
use nalgebra::{
    Point3, Rotation3, SimdBool, SimdRealField, SimdValue, UnitVector3, Vector2, Vector3,
};
use numpy::PyArray1;
use pyo3::proc_macro::{pyclass, pymethods};
use pyo3::PyResult;
use rand::Rng;

#[pyclass(name = "Sphere")]
#[derive(Debug, Clone)]
pub struct Sphere {
    center: Point3<f32>,
    radius: f32,
}

impl Sphere {
    pub fn new(center: Point3<f32>, radius: f32) -> Self {
        Sphere { center, radius }
    }
}

pub fn sphere_uv<F>(p: &Vector3<F>) -> (F, F)
where
    F: SimdRealField,
{
    let theta = (-p[1]).simd_acos();
    let phi = (-p[2]).simd_atan2(p[0]) + F::simd_pi();
    (phi / F::simd_two_pi(), theta * F::simd_frac_1_pi())
}

impl Bounded for Sphere {
    fn bounding_box(&self, _time0: f32, _time1: f32) -> AABB {
        let radius = Vector3::from_element(self.radius);
        AABB::with_bounds(self.center - radius, self.center + radius)
    }
}

impl<F, R: Rng> Hittable<F, R> for Sphere
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    #[allow(clippy::many_single_char_names)]
    fn hit(&self, ray: &Ray<F>, t_min: F, t_max: F) -> HitRecord<F> {
        let center = Point3::splat(self.center);
        let oc: Vector3<F> = ray.origin() - center;
        let half_b = oc.dot(ray.direction());
        let c = oc.norm_squared() - F::splat(self.radius * self.radius);
        let discriminant = half_b * half_b - c;
        let mask = ray.mask() & !discriminant.is_simd_negative();
        if mask.none() {
            return Default::default();
        }
        let sqrt_d = discriminant.simd_sqrt();
        let root1 = -half_b - sqrt_d;
        let mask1 = mask & root1.simd_ge(t_min) & root1.simd_le(t_max);
        let root2 = -half_b + sqrt_d;
        let mask2 = mask & root2.simd_ge(t_min) & root2.simd_le(t_max);
        let mask = mask1 | mask2;
        if mask.none() {
            return Default::default();
        }
        let t = mask1.if_else(|| root1, || root2);
        let p = ray.at(t);
        let outward_normal = UnitVector3::new_normalize(p - center);
        let (front_face, normal) = HitRecord::face_normal(ray.direction(), outward_normal);
        let (u, v) = sphere_uv(outward_normal.as_ref());
        HitRecord {
            p,
            normal,
            t,
            uv: Vector2::new(u, v),
            front_face,
            mask,
        }
    }
    fn pdf_value(&self, origin: &Point3<F>, direction: &UnitVector3<F>, mask: F::SimdBool) -> F {
        let mask = Hittable::<F, R>::test_hit(self, origin, direction, mask).mask;
        if mask.none() {
            return F::zero();
        }
        let cos_theta_max = (F::one()
            - F::splat(self.radius * self.radius)
                / (Point3::splat(self.center) - origin).norm_squared())
        .simd_sqrt();
        let solid_angle = F::simd_two_pi() * (F::one() - cos_theta_max);
        mask.if_else(
            || {
                solid_angle
                    .is_simd_positive()
                    .if_else(|| solid_angle.simd_recip(), || F::splat(f32::INFINITY))
            },
            F::zero,
        )
    }
    fn random(&self, rng: &mut R, origin: &Point3<F>) -> Vector3<F> {
        let direction = Point3::splat(self.center) - origin;
        let selector = direction.normalize()[0].simd_abs().simd_gt(F::splat(0.9));
        let up = Vector3::new(
            selector.if_else(F::zero, F::one),
            selector.if_else(F::one, F::zero),
            F::zero(),
        );
        let rot = Rotation3::face_towards(&direction, &up);
        rot * random_to_sphere(rng, F::splat(self.radius), direction.norm_squared())
    }
}

#[pymethods]
impl Sphere {
    #[new]
    pub fn py_new(center: PyVector3, radius: f32) -> Self {
        Self::new(Point3::new(center.0, center.1, center.2), radius)
    }
    #[getter("center")]
    pub fn py_center(&self) -> PyVector3 {
        (self.center[0], self.center[1], self.center[2])
    }
    #[getter("radius")]
    pub fn py_radius(&self) -> f32 {
        self.radius
    }
    #[name = "bounding_box"]
    pub fn py_bounding_box(&self, time0: f32, time1: f32) -> AABB {
        self.bounding_box(time0, time1)
    }
    #[name = "hit"]
    fn py_hit(
        &self,
        ray: &PyRay,
        t_min: &PyArray1<f32>,
        t_max: &PyArray1<f32>,
    ) -> PyResult<PyHitRecord> {
        let ray = Ray::<PySimd>::from(ray);
        let t_min = numpy_to_f(t_min)?;
        let t_max = numpy_to_f(t_max)?;
        let hit_record = Hittable::<PySimd, PyRng>::hit(self, &ray, t_min, t_max);
        Ok(PyHitRecord::from(&hit_record))
    }
}
