pub mod cosine;
pub mod hittables;
pub mod mixture;

use auto_impl::auto_impl;
use nalgebra::{SimdValue, UnitVector3};
use rand::Rng;

#[auto_impl(&, &mut, Box, Rc, Arc)]
pub trait Pdf<F: SimdValue, R: Rng> {
    fn value(&self, direction: &UnitVector3<F>, mask: F::SimdBool, rng: &mut R) -> F;
    fn generate(&self, rng: &mut R) -> UnitVector3<F>;
}

impl<F: SimdValue, R: Rng> Pdf<F, R> for ! {
    fn value(&self, _direction: &UnitVector3<F>, _mask: F::SimdBool, _rng: &mut R) -> F {
        unimplemented!()
    }

    fn generate(&self, _rng: &mut R) -> UnitVector3<F> {
        unimplemented!()
    }
}
