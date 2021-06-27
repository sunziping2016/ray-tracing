use crate::hittable::triangle::Triangle;
use crate::hittables::ManyHittables;
use crate::{SimdBoolField, SimdF32Field};
use itertools::izip;
use nalgebra::{Point3, Vector2, Vector3};
use rand::Rng;
use std::sync::Arc;
use std::vec::IntoIter;
#[derive(Debug, Clone)]

pub struct Mesh<'a> {
    mesh: &'a tobj::Mesh,
}

impl<'a> Mesh<'a> {
    pub fn new(mesh: &'a tobj::Mesh) -> Self {
        Mesh { mesh }
    }
}

impl<'a, F, R: Rng> ManyHittables<F, R> for Mesh<'a>
where
    F: SimdF32Field,
    F::SimdBool: SimdBoolField<F>,
{
    type HitItem = Arc<Triangle>;
    type SampleItem = Arc<Triangle>;
    type Iter = IntoIter<(Arc<Triangle>, Option<Arc<Triangle>>)>;

    fn into_hittables(self) -> Self::Iter {
        let mesh = self.mesh;
        let mut points = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut vertex_norms = if mesh.normal_indices.is_empty() {
            vec![Vector3::from_element(0f32); mesh.positions.len() / 3]
        } else {
            Vec::new()
        };
        for i in (0..mesh.indices.len()).step_by(3) {
            let get_point = |index: u32| {
                Point3::new(
                    mesh.positions[3 * index as usize],
                    mesh.positions[3 * index as usize + 1],
                    mesh.positions[3 * index as usize + 2],
                )
            };
            let point1 = get_point(mesh.indices[i]);
            let point2 = get_point(mesh.indices[i + 1]);
            let point3 = get_point(mesh.indices[i + 2]);
            points.push([point1, point2, point3]);
            if !mesh.normal_indices.is_empty() {
                let get_normal = |index: u32| {
                    Vector3::new(
                        mesh.normals[3 * index as usize],
                        mesh.normals[3 * index as usize + 1],
                        mesh.normals[3 * index as usize + 2],
                    )
                };
                let normal1 = get_normal(mesh.normal_indices[i]);
                let normal2 = get_normal(mesh.normal_indices[i + 1]);
                let normal3 = get_normal(mesh.normal_indices[i + 2]);
                normals.push([normal1, normal2, normal3]);
            } else {
                let norm = (point2 - point1).cross(&(point3 - point2)).normalize();
                vertex_norms[mesh.indices[i] as usize] += norm;
                vertex_norms[mesh.indices[i + 1] as usize] += norm;
                vertex_norms[mesh.indices[i + 2] as usize] += norm;
            }
            if !mesh.texcoord_indices.is_empty() {
                let get_uv = |index: u32| {
                    Vector2::new(
                        mesh.texcoords[2 * index as usize],
                        mesh.texcoords[2 * index as usize + 1],
                    )
                };
                let uv1 = get_uv(mesh.texcoord_indices[i]);
                let uv2 = get_uv(mesh.texcoord_indices[i + 1]);
                let uv3 = get_uv(mesh.texcoord_indices[i + 2]);
                uvs.push([uv1, uv2, uv3])
            } else {
                uvs.push([Vector2::new(0., 0.); 3])
            }
        }
        if mesh.normal_indices.is_empty() {
            let vertex_norms = vertex_norms
                .into_iter()
                .map(|x| x.normalize())
                .collect::<Vec<_>>();
            for i in (0..mesh.indices.len()).step_by(3) {
                let normal1 = vertex_norms[mesh.indices[i] as usize];
                let normal2 = vertex_norms[mesh.indices[i + 1] as usize];
                let normal3 = vertex_norms[mesh.indices[i + 2] as usize];
                normals.push([normal1, normal2, normal3]);
            }
        }
        let results = izip!(points.into_iter(), normals.into_iter(), uvs.into_iter())
            .map(|(p, n, uv)| {
                let shape = Arc::new(Triangle::new(p, n, uv));
                (shape.clone(), Some(shape.clone()))
            })
            .collect::<Vec<_>>();
        results.into_iter()
    }
}
