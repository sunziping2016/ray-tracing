use crate::bvh::aabb::AABB;
use crate::hittable::{Bounded, HitRecord, Hittable};
use crate::random::random_uniform;
use crate::ray::Ray;
use crate::{SimdBoolField, SimdF32Field, EPSILON};
use nalgebra::{SimdBool, SimdValue, UnitVector3};
use nalgebra::{Vector2, Vector3};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Triangle {
    vertices: [Vector3<f32>; 3],
    normals: [Vector3<f32>; 3],
    uvs: [Vector2<f32>; 3],
    // computed
    edge12: Vector3<f32>,
    edge13: Vector3<f32>,
}

impl Triangle {
    pub fn new(
        vertices: [Vector3<f32>; 3],
        normals: [Vector3<f32>; 3],
        uvs: [Vector2<f32>; 3],
    ) -> Self {
        Triangle {
            vertices,
            normals,
            uvs,
            edge12: vertices[1] - vertices[0],
            edge13: vertices[2] - vertices[0],
        }
    }
}

impl Bounded for Triangle {
    fn bounding_box(&self, _time0: f32, _time1: f32) -> AABB {
        let (min, max) = self.vertices[0].inf_sup(&self.vertices[1]);
        let (mut min, mut max) = (min.inf(&self.vertices[2]), max.sup(&self.vertices[2]));
        let diff = min - max;
        (0..3).for_each(|index| {
            if diff[index] == 0.0 {
                min[index] -= EPSILON;
                max[index] += EPSILON;
            }
        });
        AABB::with_bounds(min, max)
    }
}

impl<F, R: Rng> Hittable<F, R> for Triangle
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    fn hit(&self, ray: &Ray<F>, t_min: F, t_max: F) -> HitRecord<F> {
        let p_vec = ray.direction().cross(&Vector3::splat(self.edge13));
        let det = Vector3::splat(self.edge12).dot(&p_vec);
        let mask = ray.mask() & det.simd_abs().is_simd_positive();
        if mask.none() {
            return HitRecord::default();
        }
        let inv_det = det.simd_recip();
        let t_vec = ray.origin() - Vector3::splat(self.vertices[0]);
        let u = inv_det * t_vec.dot(&p_vec);
        let mask = mask & !u.is_simd_negative() & u.simd_le(F::one());
        if mask.none() {
            return HitRecord::default();
        }
        let q_vec = t_vec.cross(&Vector3::splat(self.edge12));
        let v = inv_det * ray.direction().dot(&q_vec);
        let mask = mask & !v.is_simd_negative() & (u + v).simd_le(F::one());
        if mask.none() {
            return HitRecord::default();
        }
        let t = inv_det * Vector3::splat(self.edge13).dot(&q_vec);
        let mask = mask & t.simd_ge(t_min) & t.simd_le(t_max);
        if mask.none() {
            return HitRecord::default();
        }
        HitRecord {
            p: ray.origin() + ray.direction().scale(t),
            normal: UnitVector3::new_normalize(
                Vector3::splat(self.normals[0]).scale(F::one() - u - v)
                    + Vector3::splat(self.normals[1]).scale(u)
                    + Vector3::splat(self.normals[2]).scale(v),
            ),
            t,
            uv: Vector2::splat(self.uvs[0]).scale(F::one() - u - v)
                + Vector2::splat(self.uvs[1]).scale(u)
                + Vector2::splat(self.uvs[2]).scale(v),
            front_face: det.is_simd_positive(), // clock-wise
            mask,
        }
    }

    fn pdf_value(&self, origin: &Vector3<F>, direction: &UnitVector3<F>, mask: F::SimdBool) -> F {
        let hit_record = Hittable::<F, R>::test_hit(self, origin, direction, mask);
        if hit_record.mask.none() {
            return F::zero();
        }
        let area = self.edge12.cross(&self.edge13).norm() * 0.5f32;
        let distance_squared = hit_record.t * hit_record.t;
        let cosine = direction.dot(&hit_record.normal).simd_abs();
        mask.if_else(
            || {
                cosine.is_simd_positive().if_else(
                    || distance_squared / (cosine * F::splat(area)),
                    || F::splat(f32::INFINITY),
                )
            },
            F::zero,
        )
    }

    fn random(&self, rng: &mut R, _origin: &Vector3<F>) -> Vector3<F> {
        let x = random_uniform::<F, _, _>(EPSILON..(1f32 - EPSILON), rng);
        let y = random_uniform::<F, _, _>(EPSILON..(1f32 - EPSILON), rng);
        let mask = (x + y).simd_gt(F::one());
        let x = mask.if_else(|| F::one() - F::splat(EPSILON) - x, || x);
        let y = mask.if_else(|| F::one() - F::splat(EPSILON) - y, || y);
        Vector3::splat(self.edge12).scale(x)
            + Vector3::splat(self.edge13).scale(y)
            + Vector3::splat(self.vertices[0])
    }
}
