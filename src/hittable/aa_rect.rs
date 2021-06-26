use crate::bvh::aabb::AABB;
use crate::hittable::{Bounded, HitRecord, Hittable};
use crate::random::random_uniform;
use crate::ray::Ray;
use crate::{SimdBoolField, SimdF32Field, EPSILON};
use nalgebra::{Point3, SimdBool, SimdValue, UnitVector3, Vector2, Vector3};
use rand::Rng;

macro_rules! rect_norm {
    ($positive:expr, 0 1 2) => {
        Vector3::new(
            F::zero(),
            F::zero(),
            if $positive { F::one() } else { -F::one() },
        )
    };
    ($positive:expr, 1 2 0) => {
        Vector3::new(
            if $positive { F::one() } else { -F::one() },
            F::zero(),
            F::zero(),
        )
    };
    ($positive:expr, 2 0 1) => {
        Vector3::new(
            F::zero(),
            if $positive { F::one() } else { -F::one() },
            F::zero(),
        )
    };
}

macro_rules! rect_bounded {
    ($self:ident $a0:ident $a1:ident $b0:ident $b1:ident 0 1 2) => {
        AABB::with_bounds(
            Point3::new($self.$a0, $self.$b0, $self.k - EPSILON),
            Point3::new($self.$a1, $self.$b1, $self.k + EPSILON),
        )
    };
    ($self:ident $a0:ident $a1:ident $b0:ident $b1:ident 1 2 0) => {
        AABB::with_bounds(
            Point3::new($self.k - EPSILON, $self.$a0, $self.$b0),
            Point3::new($self.k + EPSILON, $self.$a1, $self.$b1),
        )
    };
    ($self:ident $a0:ident $a1:ident $b0:ident $b1:ident 2 0 1) => {
        AABB::with_bounds(
            Point3::new($self.$b0, $self.k - EPSILON, $self.$a0),
            Point3::new($self.$b1, $self.k + EPSILON, $self.$a1),
        )
    };
}

macro_rules! rect_random {
    ($self:ident $rng:ident $a0:ident $a1:ident $b0:ident $b1:ident 0 1 2) => {
        Point3::new(
            random_uniform::<F, _, _>($self.$a0..=$self.$a1, $rng),
            random_uniform::<F, _, _>($self.$b0..=$self.$b1, $rng),
            F::splat($self.k),
        )
    };
    ($self:ident $rng:ident $a0:ident $a1:ident $b0:ident $b1:ident 1 2 0) => {
        Point3::new(
            F::splat($self.k),
            random_uniform::<F, _, _>($self.$a0..=$self.$a1, $rng),
            random_uniform::<F, _, _>($self.$b0..=$self.$b1, $rng),
        )
    };
    ($self:ident $rng:ident $a0:ident $a1:ident $b0:ident $b1:ident 2 0 1) => {
        Point3::new(
            random_uniform::<F, _, _>($self.$b0..=$self.$b1, $rng),
            F::splat($self.k),
            random_uniform::<F, _, _>($self.$a0..=$self.$a1, $rng),
        )
    };
}

macro_rules! rect_shape {
    ($ty:ident $a0:ident $a1:ident $b0:ident $b1:ident $idx0:tt $idx1:tt $idx2:tt) => {
        #[derive(Debug, Clone)]
        pub struct $ty {
            $a0: f32,
            $a1: f32,
            $b0: f32,
            $b1: f32,
            k: f32,
            positive: bool,
        }

        impl $ty {
            pub fn new($a0: f32, $a1: f32, $b0: f32, $b1: f32, k: f32, positive: bool) -> Self {
                Self {
                    $a0,
                    $a1,
                    $b0,
                    $b1,
                    k,
                    positive,
                }
            }
        }

        impl Bounded for $ty {
            fn bounding_box(&self, _time0: f32, _time1: f32) -> AABB {
                rect_bounded!(self $a0 $a1 $b0 $b1 $idx0 $idx1 $idx2)
            }
        }

        impl<F, R: Rng> Hittable<F, R> for $ty
        where
            F: SimdF32Field,
            F::SimdBool: SimdBoolField<F>,
        {
            fn hit(&self, ray: &Ray<F>, t_min: F, t_max: F) -> HitRecord<F> {
                let t = (F::splat(self.k) - ray.origin()[$idx2]) / ray.direction()[$idx2];
                let mask = ray.mask() & t.simd_ge(t_min) & t.simd_le(t_max);
                if mask.none() {
                    return Default::default();
                }
                let a = ray.origin()[$idx0] + t * ray.direction()[$idx0];
                let b = ray.origin()[$idx1] + t * ray.direction()[$idx1];
                let mask = mask
                    & a.simd_ge(F::splat(self.$a0))
                    & a.simd_le(F::splat(self.$a1))
                    & b.simd_ge(F::splat(self.$b0))
                    & b.simd_le(F::splat(self.$b1));
                if mask.none() {
                    return Default::default();
                }
                let u = (a - F::splat(self.$a0)) / F::splat(self.$a1 - self.$a0);
                let v = (b - F::splat(self.$b0)) / F::splat(self.$b1 - self.$b0);
                let outward_normal = UnitVector3::new_unchecked(
                    rect_norm!(self.positive, $idx0 $idx1 $idx2)
                );
                let (front_face, normal) = HitRecord::face_normal(ray.direction(), outward_normal);
                HitRecord {
                    p: ray.at(t),
                    normal,
                    t,
                    uv: Vector2::new(u, v),
                    front_face,
                    mask,
                }
            }

            fn pdf_value(
                &self,
                origin: &Point3<F>,
                direction: &UnitVector3<F>,
                mask: <F as SimdValue>::SimdBool,
            ) -> F {
                let hit_record = Hittable::<F, R>::test_hit(self, origin, direction, mask);
                if hit_record.mask.none() {
                    return F::zero();
                }
                let area = (self.$a1 - self.$a0) * (self.$b1 - self.$b0);
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

            fn random(&self, rng: &mut R, origin: &Point3<F>) -> Vector3<F> {
                let random_point = rect_random!(self rng $a0 $a1 $b0 $b1 $idx0 $idx1 $idx2);
                random_point - origin
            }
        }
    };
}

rect_shape!(XYRect x0 x1 y0 y1 0 1 2);
rect_shape!(YZRect y0 y1 z0 z1 1 2 0);
rect_shape!(ZXRect z0 z1 x0 x1 2 0 1);
