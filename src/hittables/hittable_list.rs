use crate::bvh::aabb::AABB;
use crate::hittable::{Bounded, HitRecord, Hittable};
use crate::ray::Ray;
use crate::{SimdBoolField, SimdF32Field};
use nalgebra::SimdBool;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct HittableList<O> {
    objects: Vec<O>,
}

impl<O> HittableList<O> {
    pub fn new(hittables: Vec<O>) -> Self {
        HittableList { objects: hittables }
    }
    pub fn bounding_box(&self, time0: f32, time1: f32) -> AABB
    where
        O: Bounded,
    {
        self.objects
            .iter()
            .map(|x| x.bounding_box(time0, time1))
            .reduce(|x, y| x.join(&y))
            .unwrap()
    }
    pub fn hit<F, R: Rng>(&self, ray: &Ray<F>, t_min: F, t_max: F) -> HitRecord<F>
    where
        O: Hittable<F, R>,
        F: SimdF32Field,
        F::SimdBool: SimdBoolField<F>,
    {
        let mut hit_record = HitRecord::default();
        let mut closest_so_far = t_max;
        for o in self.objects.iter() {
            let new_hr = o.hit(ray, t_min, closest_so_far);
            closest_so_far = new_hr.mask.if_else(|| new_hr.t, || closest_so_far);
            hit_record = new_hr.mask.if_else(|| new_hr, || hit_record)
        }
        hit_record
    }
}
