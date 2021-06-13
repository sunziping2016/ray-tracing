use crate::simd::MyFromElement;
use nalgebra::{ClosedSub, Scalar, SimdPartialOrd, SimdValue, Vector3};
use num_traits::Float;

#[derive(Debug, Clone)]
pub struct AABB<R: SimdValue> {
    pub min: Vector3<R>,
    pub max: Vector3<R>,
    pub mask: R::SimdBool,
}

pub trait Bounded {
    type Field: SimdValue;

    fn aabb(&self) -> AABB<Self::Field>;
}

impl<R: SimdValue> AABB<R> {
    pub fn with_bounds(min: Vector3<R>, max: Vector3<R>, mask: R::SimdBool) -> Self {
        Self { min, max, mask }
    }
    pub fn empty(mask: R::SimdBool) -> Self
    where
        R: MyFromElement + Copy,
        R::Element: Float,
    {
        let infinity = R::from_element(R::Element::infinity());
        let neg_infinity = R::from_element(R::Element::neg_infinity());
        Self {
            min: Vector3::new(infinity, infinity, infinity),
            max: Vector3::new(neg_infinity, neg_infinity, neg_infinity),
            mask,
        }
    }
    pub fn join(&self, other: &Self) -> Self
    where
        R: SimdPartialOrd + MyFromElement + Copy + Scalar,
        R::Element: Float,
    {
        let infinity = R::from_element(R::Element::infinity());
        let neg_infinity = R::from_element(R::Element::neg_infinity());
        let min = Vector3::new(
            self.min[0]
                .select(self.mask, infinity)
                .simd_min(other.min[0].select(other.mask, infinity)),
            self.min[1]
                .select(self.mask, infinity)
                .simd_min(other.min[1].select(other.mask, infinity)),
            self.min[2]
                .select(self.mask, infinity)
                .simd_min(other.min[2].select(other.mask, infinity)),
        );
        let max = Vector3::new(
            self.max[0]
                .select(self.mask, neg_infinity)
                .simd_max(other.max[0].select(other.mask, neg_infinity)),
            self.max[1]
                .select(self.mask, neg_infinity)
                .simd_max(other.max[1].select(other.mask, neg_infinity)),
            self.max[2]
                .select(self.mask, neg_infinity)
                .simd_max(other.max[2].select(other.mask, neg_infinity)),
        );
        let mask = self.mask | other.mask;
        Self { min, max, mask }
    }
    pub fn grow(&self, other: Vector3<R>, mask: R::SimdBool) -> Self
    where
        R: SimdPartialOrd + MyFromElement + Copy + Scalar,
        R::Element: Float,
    {
        let infinity = R::from_element(R::Element::infinity());
        let neg_infinity = R::from_element(R::Element::neg_infinity());
        let min = Vector3::new(
            self.min[0]
                .select(self.mask, infinity)
                .simd_min(other[0].select(mask, infinity)),
            self.min[1]
                .select(self.mask, infinity)
                .simd_min(other[1].select(mask, infinity)),
            self.min[2]
                .select(self.mask, infinity)
                .simd_min(other[2].select(mask, infinity)),
        );
        let max = Vector3::new(
            self.max[0]
                .select(self.mask, neg_infinity)
                .simd_max(other[0].select(mask, neg_infinity)),
            self.max[1]
                .select(self.mask, neg_infinity)
                .simd_max(other[1].select(mask, neg_infinity)),
            self.max[2]
                .select(self.mask, neg_infinity)
                .simd_max(other[2].select(mask, neg_infinity)),
        );
        let mask = self.mask | mask;
        Self { min, max, mask }
    }
    pub fn size(&self) -> Vector3<R>
    where
        R: Scalar + ClosedSub + Copy,
    {
        self.max - self.min
    }
    // pub fn contains(&self, p: &Vector3<R>) -> R::SimdBool {
    //     izip!(self.min.iter(), p.iter(), self.max.iter())
    //         .map(|(&min, &x, &max)| min.simd_le(x) & x.simd_le(max))
    //         .reduce(|lhs, rhs| lhs & rhs)
    //         .unwrap()
    // }
    // pub fn approx_contains_eps(&self, p: &Vector3<R>, epsilon: f32) -> R::SimdBool {
    //     let epsilon = R::from_subset(&(epsilon as f64));
    //     izip!(self.min.iter(), p.iter(), self.max.iter())
    //         .map(|(&min, &x, &max)| (x - min).simd_gt(-epsilon) & (x - max).simd_lt(epsilon))
    //         .reduce(|lhs, rhs| lhs & rhs)
    //         .unwrap()
    // }
}
