use nalgebra::Vector3;

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

impl AABB {
    pub fn with_bounds(min: Vector3<f32>, max: Vector3<f32>) -> Self {
        Self { min, max }
    }
    pub fn empty() -> Self {
        Self {
            min: Vector3::from_element(f32::INFINITY),
            max: Vector3::from_element(f32::NEG_INFINITY),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.min[0] > self.max[0] || self.min[1] > self.max[1] || self.min[2] > self.max[2]
    }
    pub fn join(&self, other: &Self) -> Self {
        let min = Vector3::new(
            self.min[0].min(other.min[0]),
            self.min[1].min(other.min[1]),
            self.min[2].min(other.min[2]),
        );
        let max = Vector3::new(
            self.max[0].max(other.max[0]),
            self.max[1].max(other.max[1]),
            self.max[2].max(other.max[2]),
        );
        Self { min, max }
    }
    pub fn grow(&self, other: &Vector3<f32>) -> Self {
        let min = Vector3::new(
            self.min[0].min(other[0]),
            self.min[1].min(other[1]),
            self.min[2].min(other[2]),
        );
        let max = Vector3::new(
            self.max[0].max(other[0]),
            self.max[1].max(other[1]),
            self.max[2].max(other[2]),
        );
        Self { min, max }
    }
    pub fn size(&self) -> Vector3<f32> {
        self.max - self.min
    }
    pub fn center(&self) -> Vector3<f32> {
        self.min + self.size().scale(0.5)
    }
    pub fn surface_area(&self) -> f32 {
        2.0 * self.size().norm_squared()
    }
}
