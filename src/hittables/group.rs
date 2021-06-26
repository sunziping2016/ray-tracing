use crate::bvh::aabb::AABB;
use crate::hittable::{Bounded, HitRecord, Hittable, Samplable};
use crate::ray::Ray;
use crate::{SimdBoolField, SimdF32Field};
use nalgebra::{Point3, SimdBool, SimdValue, UnitVector3};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct HittableGroup<O> {
    objects: Vec<O>,
}

impl<O> HittableGroup<O> {
    pub fn new() -> Self {
        HittableGroup {
            objects: Vec::new(),
        }
    }
    pub fn add(&mut self, hittable: O) {
        self.objects.push(hittable)
    }
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
    pub fn len(&self) -> usize {
        self.objects.len()
    }
}

impl<O> Bounded for HittableGroup<O>
where
    O: Bounded,
{
    fn bounding_box(&self, time0: f32, time1: f32) -> AABB {
        self.objects
            .iter()
            .map(|x| x.bounding_box(time0, time1))
            .reduce(|x, y| x.join(&y))
            .unwrap()
    }
}

impl<O, F, R: Rng> Hittable<F, R> for HittableGroup<O>
where
    O: Hittable<F, R>,
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    fn hit(&self, ray: &Ray<F>, t_min: F, t_max: F, rng: &mut R) -> HitRecord<F> {
        let mut hit_record = HitRecord::default();
        let mut closest_so_far = t_max;
        for o in self.objects.iter() {
            let new_hr = o.hit(ray, t_min, closest_so_far, rng);
            closest_so_far = new_hr.mask.if_else(|| new_hr.t, || closest_so_far);
            hit_record = new_hr.mask.if_else(|| new_hr, || hit_record)
        }
        hit_record
    }
}

impl<O, F, R: Rng> Samplable<F, R> for HittableGroup<O>
where
    O: Samplable<F, R>,
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    fn value(
        &self,
        origin: &Point3<F>,
        direction: &UnitVector3<F>,
        mask: <F as SimdValue>::SimdBool,
        rng: &mut R,
    ) -> F {
        let weight = 1f32 / self.objects.len() as f32;
        let mut sum = F::zero();
        for object in self.objects.iter() {
            sum += F::splat(weight) * object.value(origin, direction, mask, rng);
        }
        sum
    }

    fn generate(&self, origin: &Point3<F>, rng: &mut R) -> UnitVector3<F> {
        // FIXME: batch random?
        let index = rng.gen_range(0..self.objects.len());
        self.objects[index].generate(origin, rng)
    }
}
