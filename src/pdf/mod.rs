pub mod cosine;

use nalgebra::UnitVector3;
use rand::Rng;

pub trait Pdf<F> {
    fn value(&self, direction: &UnitVector3<F>) -> F;
    fn generate<R: Rng>(&self, rng: &mut R) -> UnitVector3<F>;
}
