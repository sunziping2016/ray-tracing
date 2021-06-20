pub mod cosine;

use nalgebra::UnitVector3;
use rand::Rng;

pub trait Pdf<F, R: Rng> {
    fn value(&self, direction: &UnitVector3<F>) -> F;
    fn generate(&self, rng: &mut R) -> UnitVector3<F>;
}

impl<F, R: Rng> Pdf<F, R> for ! {
    fn value(&self, _direction: &UnitVector3<F>) -> F {
        unimplemented!()
    }

    fn generate(&self, _rng: &mut R) -> UnitVector3<F> {
        unimplemented!()
    }
}
