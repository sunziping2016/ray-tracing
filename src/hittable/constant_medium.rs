use crate::bvh::aabb::AABB;
use crate::hittable::{Bounded, HitRecord, Hittable};
use crate::hittables::hittable_list::HittableList;
use crate::random::random_uniform;
use crate::ray::Ray;
use crate::{SimdBoolField, SimdF32Field, EPSILON};
use nalgebra::{SimdBool, SimdValue, UnitVector3, Vector3};
use num_traits::Zero;
use rand::{thread_rng, Rng};

#[derive(Debug, Clone)]
pub struct ConstantMedium<O> {
    hittable: HittableList<O>,
    neg_inv_density: f32,
}

impl<O> ConstantMedium<O> {
    pub fn new(hittable: HittableList<O>, density: f32) -> Self {
        ConstantMedium {
            hittable,
            neg_inv_density: -1.0 / density,
        }
    }
}

impl<O> Bounded for ConstantMedium<O>
where
    O: Bounded,
{
    fn bounding_box(&self, time0: f32, time1: f32) -> AABB {
        self.hittable.bounding_box(time0, time1)
    }
}

impl<O, F, R: Rng> Hittable<F, R> for ConstantMedium<O>
where
    O: Hittable<F, R>,
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    fn hit(&self, ray: &Ray<F>, t_min: F, t_max: F) -> HitRecord<F> {
        let hit_record1 =
            self.hittable
                .hit(ray, F::splat(f32::NEG_INFINITY), F::splat(f32::INFINITY));
        let mask = ray.mask() & hit_record1.mask;
        if mask.none() {
            return HitRecord::default();
        }
        let hit_record2 = self.hittable.hit(
            ray,
            hit_record1.t + F::splat(EPSILON),
            F::splat(f32::INFINITY),
        );
        let mask = mask & hit_record2.mask;
        if mask.none() {
            return HitRecord::default();
        }
        let t_min = hit_record1.t.simd_max(t_min);
        let t_max = hit_record2.t.simd_min(t_max);
        let mask = mask & t_min.simd_lt(t_max);
        if mask.none() {
            return HitRecord::default();
        }
        let t_min = t_min.simd_max(F::zero());
        let distance_inside_boundary = t_max - t_min;
        let hit_distance = F::splat(self.neg_inv_density)
            * random_uniform::<F, _, _>(0f32..=1f32, &mut thread_rng()).simd_ln();
        let mask = mask & hit_distance.simd_le(distance_inside_boundary);
        if mask.none() {
            return HitRecord::default();
        }
        let t = hit_record1.t + hit_distance;
        let p = ray.at(t);
        HitRecord {
            p,
            normal: UnitVector3::new_unchecked(Vector3::new(F::one(), F::zero(), F::zero())),
            t,
            uv: Zero::zero(),
            front_face: F::SimdBool::splat(true),
            mask,
        }
    }
}
