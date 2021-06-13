use nalgebra::{RealField, SimdValue, UnitVector3, Vector3};

#[derive(Debug, Clone)]
pub struct Ray<R: SimdValue> {
    origin: Vector3<R>,
    direction: UnitVector3<R>,
    time: R,
    mask: R::SimdBool,
}

impl<R: SimdValue> Ray<R> {
    pub fn new(origin: Vector3<R>, direction: UnitVector3<R>, time: R, mask: R::SimdBool) -> Self {
        Ray {
            origin,
            direction,
            time,
            mask,
        }
    }
    pub fn at(&self, t: R) -> Vector3<R>
    where
        R: RealField,
    {
        self.origin + self.direction.scale(t)
    }
}
