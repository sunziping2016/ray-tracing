use crate::bvh::aabb::AABB;
use crate::ray::Ray;
use crate::simd::MySimdVector;
use crate::EPSILON;
use nalgebra::{Point3, SimdBool, SimdRealField, SimdValue};
use std::fmt::Debug;

#[derive(Debug, Clone)]
enum BVHVariants {
    Leaf {
        shape_index: usize,
    },
    Node {
        children_index: [usize; 2],
        children_aabb: [AABB; 2],
    },
}

#[derive(Debug, Clone)]
struct BVHNode {
    parent_index: usize,
    depth: u32,
    variant: BVHVariants,
}

impl BVHNode {
    fn create_dummy() -> BVHNode {
        BVHNode {
            parent_index: 0,
            depth: 0,
            variant: BVHVariants::Leaf { shape_index: 0 },
        }
    }
    fn build(
        shapes: &[AABB],
        shape_centroids: &[Point3<f32>],
        indices: &[usize],
        parent_index: usize,
        depth: u32,
        nodes: &mut Vec<BVHNode>,
        node_indices: &mut [usize],
    ) -> usize {
        if indices.len() == 1 {
            let shape_index = indices[0];
            let node_index = nodes.len();
            nodes.push(BVHNode {
                parent_index,
                depth,
                variant: BVHVariants::Leaf { shape_index },
            });
            node_indices[shape_index] = node_index;
            return node_index;
        }
        let centroid_bounds = indices
            .iter()
            .map(|&index| &shape_centroids[index])
            .fold(AABB::empty(), |centroids, centroid| {
                centroids.grow(centroid)
            });
        let node_index = nodes.len();
        nodes.push(BVHNode::create_dummy());
        let (split_axis, split_axis_size) = centroid_bounds.size().argmax();
        let (children_index, children_aabb) = if split_axis_size < EPSILON {
            let (child_l_indices, child_r_indices) = indices.split_at(indices.len() / 2);
            let mut join_shapes = |indices: &[usize]| {
                let aabb = indices
                    .iter()
                    .map(|&index| &shapes[index])
                    .fold(AABB::empty(), |aabbs, aabb| aabbs.join(aabb));
                let index = Self::build(
                    shapes,
                    shape_centroids,
                    indices,
                    node_index,
                    depth + 1,
                    nodes,
                    node_indices,
                );
                (aabb, index)
            };
            let (child_l_aabb, child_l_index) = join_shapes(child_l_indices);
            let (child_r_aabb, child_r_index) = join_shapes(child_r_indices);
            ([child_l_index, child_r_index], [child_l_aabb, child_r_aabb])
        } else {
            const NUM_BUCKETS: usize = 6;
            let mut buckets = [(0, AABB::empty()); NUM_BUCKETS];
            let mut bucket_assignments: [Vec<usize>; NUM_BUCKETS] = Default::default();

            indices
                .iter()
                .map(|&index| (index, &shapes[index], &shape_centroids[index]))
                .for_each(|(index, aabb, center)| {
                    let relative =
                        (center[split_axis] - centroid_bounds.min[split_axis]) / split_axis_size;
                    let bucket_num = (relative * (NUM_BUCKETS as f32 - 0.01)) as usize;
                    buckets[bucket_num].0 += 1;
                    buckets[bucket_num].1 = buckets[bucket_num].1.join(aabb);
                    bucket_assignments[bucket_num].push(index)
                });

            let mut min_bucket = 0;
            let mut min_cost = f32::INFINITY;
            let mut child_l_aabb = AABB::empty();
            let mut child_r_aabb = AABB::empty();
            for i in 0..(NUM_BUCKETS - 1) {
                let (l_buckets, r_buckets) = buckets.split_at(i + 1);
                let join_bucket = |buckets: &[(usize, AABB)]| {
                    buckets
                        .iter()
                        .fold((0, AABB::empty()), |(a_size, a_aabb), (b_size, b_aabb)| {
                            (a_size + b_size, a_aabb.join(b_aabb))
                        })
                };
                let child_l = join_bucket(l_buckets);
                let child_r = join_bucket(r_buckets);
                let cost = child_l.0 as f32 * child_l.1.surface_area()
                    + child_r.0 as f32 * child_r.1.surface_area();
                if cost < min_cost {
                    min_bucket = i;
                    min_cost = cost;
                    child_l_aabb = child_l.1;
                    child_r_aabb = child_r.1;
                }
            }
            let (l_assignments, r_assignments) = bucket_assignments.split_at(min_bucket + 1);
            let mut build = |assignments: &[Vec<usize>]| {
                let indices = assignments.iter().flatten().copied().collect::<Vec<_>>();
                Self::build(
                    shapes,
                    shape_centroids,
                    &indices,
                    node_index,
                    depth + 1,
                    nodes,
                    node_indices,
                )
            };
            let child_l_index = build(l_assignments);
            let child_r_index = build(r_assignments);
            ([child_l_index, child_r_index], [child_l_aabb, child_r_aabb])
        };
        assert!(!children_aabb[0].is_empty());
        assert!(!children_aabb[1].is_empty());
        nodes[node_index] = BVHNode {
            parent_index,
            depth,
            variant: BVHVariants::Node {
                children_aabb,
                children_index,
            },
        };
        node_index
    }
}

#[derive(Debug, Clone)]
pub struct BVH {
    nodes: Vec<BVHNode>,
}

impl BVH {
    pub fn build(shapes: &[AABB]) -> Self {
        if shapes.is_empty() {
            return Self { nodes: Vec::new() };
        }
        let shape_centroids = shapes.iter().map(|x| x.center()).collect::<Vec<_>>();
        let indices = (0..shapes.len()).collect::<Vec<_>>();
        let mut nodes = Vec::with_capacity(shapes.len() * 2);
        let mut node_indices = vec![0; shapes.len()];
        BVHNode::build(
            shapes,
            &shape_centroids,
            &indices,
            0,
            0,
            &mut nodes,
            &mut node_indices,
        );
        Self { nodes }
    }
    #[allow(clippy::uninit_assumed_init)]
    pub fn traverse<F: SimdValue>(&self, ray: &Ray<F>, t_min: F, t_max: F) -> [Vec<usize>; F::LANES]
    where
        F: SimdRealField<Element = f32> + MySimdVector,
        [Vec<usize>; F::LANES]: Sized,
    {
        fn traverse_recursive<F: SimdRealField<Element = f32> + MySimdVector>(
            nodes: &[BVHNode],
            node_index: usize,
            ray: &Ray<F>,
            mask: F::SimdBool,
            t_min: F,
            t_max: F,
            indices: &mut [Vec<usize>; F::LANES],
        ) {
            match nodes[node_index].variant {
                BVHVariants::Node {
                    children_aabb,
                    children_index,
                } => {
                    let mut build = |index: usize| {
                        let new_mask = ray.intersects_aabb(&children_aabb[index], t_min, t_max);
                        if new_mask.any() {
                            traverse_recursive(
                                nodes,
                                children_index[index],
                                ray,
                                new_mask,
                                t_min,
                                t_max,
                                indices,
                            );
                        }
                    };
                    build(0);
                    build(1);
                }
                BVHVariants::Leaf { shape_index, .. } => {
                    let bits = mask.bitmask();
                    for (index, indices) in indices.iter_mut().enumerate() {
                        if bits & (1 << index) != 0 {
                            indices.push(shape_index);
                        }
                    }
                }
            }
        }
        let mut indices: [Vec<usize>; F::LANES] = unsafe {
            let mut arr: [Vec<usize>; F::LANES] = std::mem::MaybeUninit::uninit().assume_init();
            for item in arr.iter_mut() {
                std::ptr::write(item, Vec::new());
            }
            arr
        };
        if ray.mask().any() && !self.nodes.is_empty() {
            traverse_recursive(&self.nodes, 0, ray, ray.mask(), t_min, t_max, &mut indices);
        }
        indices
    }
}
