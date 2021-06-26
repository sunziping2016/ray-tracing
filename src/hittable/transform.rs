use crate::bvh::aabb::AABB;
use crate::hittable::{Bounded, HitRecord, Hittable};
use crate::ray::Ray;
use crate::SimdF32Field;
use nalgebra::{Matrix3, Point3, Projective3, SimdValue, UnitVector3, Vector3};
use rand::Rng;

pub struct TransformHittable<O> {
    transform: Matrix3<f32>,
    translation: Vector3<f32>,
    inv_transform: Matrix3<f32>,
    inv_translation: Vector3<f32>,
    object: O,
}

impl<O> TransformHittable<O> {
    pub fn new(projective: Projective3<f32>, object: O) -> Self {
        let inv = projective.inverse();
        let transform = Matrix3::from(projective.matrix().fixed_slice::<3, 3>(0, 0));
        let translation = Vector3::from(projective.matrix().fixed_slice::<3, 1>(0, 3));
        let inv_transform = Matrix3::from(inv.matrix().fixed_slice::<3, 3>(0, 0));
        let inv_translation = Vector3::from(inv.matrix().fixed_slice::<3, 1>(0, 3));
        TransformHittable {
            transform,
            translation,
            inv_transform,
            inv_translation,
            object,
        }
    }
}

impl<O> Bounded for TransformHittable<O>
where
    O: Bounded,
{
    fn bounding_box(&self, time0: f32, time1: f32) -> AABB {
        let AABB {
            min: origin_min,
            max: origin_max,
        } = self.object.bounding_box(time0, time1);
        let vertices = [
            Point3::new(origin_min[0], origin_min[1], origin_min[2]),
            Point3::new(origin_min[0], origin_min[1], origin_max[2]),
            Point3::new(origin_min[0], origin_max[1], origin_min[2]),
            Point3::new(origin_min[0], origin_max[1], origin_max[2]),
            Point3::new(origin_max[0], origin_min[1], origin_min[2]),
            Point3::new(origin_max[0], origin_min[1], origin_max[2]),
            Point3::new(origin_max[0], origin_max[1], origin_min[2]),
            Point3::new(origin_max[0], origin_max[1], origin_max[2]),
        ];
        let vertices = vertices.map(|v| self.transform * v + self.translation);
        let min = vertices.iter().copied().reduce(|x, y| x.inf(&y)).unwrap();
        let max = vertices.iter().copied().reduce(|x, y| x.sup(&y)).unwrap();
        AABB { min, max }
    }
}

impl<O, F, R: Rng> Hittable<F, R> for TransformHittable<O>
where
    O: Hittable<F, R>,
    F: SimdF32Field,
{
    fn hit(&self, ray: &Ray<F>, t_min: F, t_max: F) -> HitRecord<F> {
        let transform = Matrix3::<F>::splat(self.transform);
        let translation = Vector3::<F>::splat(self.translation);
        let inv_transform = Matrix3::<F>::splat(self.inv_transform);
        let inv_translation = Vector3::<F>::splat(self.inv_translation);
        let direction = inv_transform * ray.direction().as_ref();
        let norm = direction.norm();
        let ray = Ray::new(
            inv_transform * ray.origin() + inv_translation,
            UnitVector3::new_unchecked(direction.unscale(norm)),
            *ray.time(),
            ray.mask(),
        );
        let HitRecord {
            p,
            normal,
            t,
            uv,
            front_face,
            mask,
        } = self.object.hit(&ray, t_min * norm, t_max * norm);
        let p = transform * p + translation;
        let normal = transform * normal.as_ref();
        HitRecord {
            p,
            normal: UnitVector3::new_normalize(normal),
            t: t / norm,
            uv,
            front_face,
            mask,
        }
    }

    fn pdf_value(&self, origin: &Point3<F>, direction: &UnitVector3<F>, mask: F::SimdBool) -> F {
        let inv_transform = Matrix3::<F>::splat(self.inv_transform);
        let inv_translation = Vector3::<F>::splat(self.inv_translation);
        let origin = inv_transform * origin + inv_translation;
        let direction = inv_transform * direction.as_ref();
        self.object
            .pdf_value(&origin, &UnitVector3::new_normalize(direction), mask)
    }

    fn random(&self, rng: &mut R, origin: &Point3<F>) -> Vector3<F> {
        let transform = Matrix3::<F>::splat(self.transform);
        let inv_transform = Matrix3::<F>::splat(self.inv_transform);
        let inv_translation = Vector3::<F>::splat(self.inv_translation);
        let origin = inv_transform * origin + inv_translation;
        transform * self.object.random(rng, &origin)
    }
}
