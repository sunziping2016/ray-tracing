use crate::camera::CameraParam;
use crate::hittables::ManyHittables;
use crate::renderer::RendererParam;
use crate::scene::Scene;
use crate::{hittable, hittables, material, texture, SimdBoolField, SimdF32Field};
use crate::{BoxedHittable, BoxedMaterial, BoxedSamplable, BoxedTexture};
use image::io::Reader;
use nalgebra::{Matrix4, Point3, Projective3, Vector2, Vector3};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XYRect {
    pub x0: f32,
    pub x1: f32,
    pub y0: f32,
    pub y1: f32,
    pub z: f32,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub positive: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YZRect {
    pub y0: f32,
    pub y1: f32,
    pub z0: f32,
    pub z1: f32,
    pub x: f32,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub positive: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZXRect {
    pub z0: f32,
    pub z1: f32,
    pub x0: f32,
    pub x1: f32,
    pub y: f32,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub positive: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantMedium {
    pub shape: Box<NameOrShape>,
    pub density: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triangle {
    pub vertices: [[f32; 3]; 3],
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub normals: Option<[[f32; 3]; 3]>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub uvs: Option<[[f32; 2]; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cuboid {
    p0: [f32; 3],
    p1: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum IntOrString {
    Int(usize),
    String(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mesh {
    file: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model: Option<IntOrString>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "kebab-case")]
pub enum AnyShape {
    Sphere(Sphere),
    #[serde(rename = "xy-rect")]
    XYRect(XYRect),
    #[serde(rename = "yz-rect")]
    YZRect(YZRect),
    #[serde(rename = "zx-rect")]
    ZXRect(ZXRect),
    ConstantMedium(ConstantMedium),
    Triangle(Triangle),
    Cuboid(Cuboid),
    Mesh(Mesh),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shape {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    #[serde(flatten)]
    pub shape: AnyShape,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub transform: Option<[[f32; 3]; 3]>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub translate: Option<[f32; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NameOrShape {
    Name(String),
    Shape(Shape),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolidColor {
    color: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checker {
    odd: Box<NameOrTexture>,
    even: Box<NameOrTexture>,
    density: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Noise {
    scale: f32,
    depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "kebab-case")]
pub enum AnyTexture {
    SolidColor(SolidColor),
    Checker(Checker),
    Image(Image),
    Noise(Noise),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Texture {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    #[serde(flatten)]
    pub texture: AnyTexture,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NameOrTexture {
    Name(String),
    Texture(Texture),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lambertian {
    texture: NameOrTexture,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Isotropic {
    albedo: NameOrTexture,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dielectric {
    ir: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffuseLight {
    emit: NameOrTexture,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metal {
    albedo: [f32; 3],
    fuzz: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "kebab-case")]
pub enum AnyMaterial {
    Lambertian(Lambertian),
    Isotropic(Isotropic),
    Dielectric(Dielectric),
    DiffuseLight(DiffuseLight),
    Metal(Metal),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    #[serde(flatten)]
    pub material: AnyMaterial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NameOrMaterial {
    Name(String),
    Material(Material),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Object {
    pub shape: NameOrShape,
    pub material: NameOrMaterial,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub important: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub visible: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneParam {
    pub renderer: RendererParam,
    pub camera: CameraParam,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub background: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub environment: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub objects: Vec<Object>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub shapes: Vec<Shape>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub materials: Vec<Material>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub textures: Vec<Texture>,
}

struct VisitContext<'a, F, R: Rng> {
    scene: &'a mut Scene<F, R>,
    shapes: HashMap<String, Vec<(BoxedHittable<F, R>, Option<BoxedSamplable<F, R>>)>>,
    materials: HashMap<String, BoxedMaterial<F, R>>,
    textures: HashMap<String, BoxedTexture<F>>,
    visiting_shapes: HashSet<String>,
    visiting_materials: HashSet<String>,
    visiting_textures: HashSet<String>,
    name_shapes: HashMap<String, &'a Shape>,
    name_materials: HashMap<String, &'a Material>,
    name_textures: HashMap<String, &'a Texture>,
}

impl<'a, F, R: Rng> VisitContext<'a, F, R> {
    fn new(scene: &'a mut Scene<F, R>, param: &'a SceneParam) -> Self {
        Self {
            scene,
            shapes: HashMap::new(),
            materials: HashMap::new(),
            textures: HashMap::new(),
            visiting_shapes: HashSet::new(),
            visiting_materials: HashSet::new(),
            visiting_textures: HashSet::new(),
            name_shapes: param
                .shapes
                .iter()
                .filter_map(|x| {
                    if let Some(name) = &x.name {
                        Some((name.clone(), x))
                    } else {
                        None
                    }
                })
                .collect(),
            name_materials: param
                .materials
                .iter()
                .filter_map(|x| {
                    if let Some(name) = &x.name {
                        Some((name.clone(), x))
                    } else {
                        None
                    }
                })
                .collect(),
            name_textures: param
                .textures
                .iter()
                .filter_map(|x| {
                    if let Some(name) = &x.name {
                        Some((name.clone(), x))
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }
}

fn visit_texture<F, R: Rng>(context: &mut VisitContext<F, R>, text: &Texture) -> BoxedTexture<F>
where
    F: SimdF32Field + Into<[f32; F::LANES]> + From<[f32; F::LANES]>,
    F::SimdBool: SimdBoolField<F>,
{
    if let Some(name) = text.name.as_ref() {
        if let Some(result) = context.textures.get(name) {
            return result.clone();
        }
        assert!(!context.visiting_textures.contains(name));
        context.visiting_textures.insert(name.clone());
    }
    let texture = match &text.texture {
        AnyTexture::SolidColor(solid_color) => {
            Arc::new(texture::solid_color::SolidColor::new(Vector3::new(
                solid_color.color[0],
                solid_color.color[1],
                solid_color.color[2],
            ))) as BoxedTexture<F>
        }
        AnyTexture::Checker(checker) => Arc::new(texture::checker::Checker::new(
            visit_texture(
                context,
                match &*checker.odd {
                    NameOrTexture::Name(name) => context.name_textures[name],
                    NameOrTexture::Texture(text) => text,
                },
            ),
            visit_texture(
                context,
                match &*checker.even {
                    NameOrTexture::Name(name) => context.name_textures[name],
                    NameOrTexture::Texture(text) => text,
                },
            ),
            checker.density,
        )),
        AnyTexture::Image(image) => Arc::new(texture::image::ImageTexture::new(
            Reader::open(&image.file)
                .expect("failed to open image")
                .decode()
                .expect("failed to decode image"),
        )),
        AnyTexture::Noise(noise) => Arc::new(texture::noise::Noise::new(
            texture::noise::Perlin::new(&mut thread_rng()),
            noise.scale,
            noise.depth,
        )),
    };
    if let Some(name) = text.name.as_ref() {
        context.visiting_textures.remove(name);
        context.textures.insert(name.clone(), texture.clone());
    }
    texture
}

fn visit_material<F, R: Rng>(
    context: &mut VisitContext<F, R>,
    mat: &Material,
) -> BoxedMaterial<F, R>
where
    F: SimdF32Field + Into<[f32; F::LANES]> + From<[f32; F::LANES]>,
    F::SimdBool: SimdBoolField<F>,
{
    if let Some(name) = mat.name.as_ref() {
        if let Some(result) = context.materials.get(name) {
            return result.clone();
        }
        assert!(!context.visiting_materials.contains(name));
        context.visiting_materials.insert(name.clone());
    }
    let material = match &mat.material {
        AnyMaterial::Lambertian(lambertian) => {
            Arc::new(material::lambertian::Lambertian::new(visit_texture(
                context,
                match &lambertian.texture {
                    NameOrTexture::Name(name) => context.name_textures[name],
                    NameOrTexture::Texture(text) => text,
                },
            ))) as BoxedMaterial<F, R>
        }
        AnyMaterial::Isotropic(isotropic) => {
            Arc::new(material::isotropic::Isotropic::new(visit_texture(
                context,
                match &isotropic.albedo {
                    NameOrTexture::Name(name) => context.name_textures[name],
                    NameOrTexture::Texture(text) => text,
                },
            )))
        }
        AnyMaterial::Dielectric(dielectric) => {
            Arc::new(material::dielectric::Dielectric::new(dielectric.ir))
        }
        AnyMaterial::DiffuseLight(light) => {
            Arc::new(material::diffuse_light::DiffuseLight::new(visit_texture(
                context,
                match &light.emit {
                    NameOrTexture::Name(name) => context.name_textures[name],
                    NameOrTexture::Texture(text) => text,
                },
            )))
        }
        AnyMaterial::Metal(metal) => Arc::new(material::metal::Metal::new(
            Vector3::new(metal.albedo[0], metal.albedo[1], metal.albedo[2]),
            metal.fuzz,
        )),
    };
    if let Some(name) = mat.name.as_ref() {
        context.visiting_materials.remove(name);
        context.materials.insert(name.clone(), material.clone());
    }
    material
}

fn visit_shape<F, R: 'static + Rng>(
    context: &mut VisitContext<F, R>,
    shape: &Shape,
) -> Vec<(BoxedHittable<F, R>, Option<BoxedSamplable<F, R>>)>
where
    F: SimdF32Field + Into<[f32; F::LANES]> + From<[f32; F::LANES]>,
    F::SimdBool: SimdBoolField<F>,
{
    if let Some(name) = shape.name.as_ref() {
        if let Some(result) = context.shapes.get(name) {
            return result.clone();
        }
        assert!(!context.visiting_shapes.contains(name));
        context.visiting_shapes.insert(name.clone());
    }
    let projection = if shape.translate.is_some() || shape.transform.is_some() {
        let transform = shape
            .transform
            .unwrap_or([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        let translation = shape.translate.unwrap_or([0., 0., 0.]);
        let projection = Projective3::from_matrix_unchecked(Matrix4::new(
            transform[0][0],
            transform[0][1],
            transform[0][2],
            translation[0],
            transform[1][0],
            transform[1][1],
            transform[1][2],
            translation[1],
            transform[2][0],
            transform[2][1],
            transform[2][2],
            translation[2],
            0.,
            0.,
            0.,
            1.,
        ));
        Some(projection)
    } else {
        None
    };
    macro_rules! wrap {
        ($expr:expr, $project:ident) => {
            if let Some(projection) = $project {
                let shape = Arc::new(hittable::transform::TransformHittable::new(
                    projection, $expr,
                ));
                vec![(
                    shape.clone() as BoxedHittable<F, R>,
                    Some(shape as BoxedSamplable<F, R>),
                )]
            } else {
                let shape = Arc::new($expr);
                vec![(
                    shape.clone() as BoxedHittable<F, R>,
                    Some(shape as BoxedSamplable<F, R>),
                )]
            }
        };
    }
    macro_rules! wrap_no_sample {
        ($expr:expr, $project:ident) => {
            if let Some(projection) = $project {
                let shape = Arc::new(hittable::transform::TransformHittable::new(
                    projection, $expr,
                ));
                vec![(shape.clone() as BoxedHittable<F, R>, None)]
            } else {
                let shape = Arc::new($expr);
                vec![(shape.clone() as BoxedHittable<F, R>, None)]
            }
        };
    }
    let result = match &shape.shape {
        AnyShape::Sphere(sphere) => {
            wrap!(
                hittable::sphere::Sphere::new(
                    Point3::new(sphere.center[0], sphere.center[1], sphere.center[2]),
                    sphere.radius,
                ),
                projection
            )
        }
        AnyShape::XYRect(rect) => {
            wrap!(
                hittable::aa_rect::XYRect::new(
                    rect.x0,
                    rect.x1,
                    rect.y0,
                    rect.y1,
                    rect.z,
                    rect.positive.unwrap_or(true),
                ),
                projection
            )
        }
        AnyShape::YZRect(rect) => {
            wrap!(
                hittable::aa_rect::YZRect::new(
                    rect.y0,
                    rect.y1,
                    rect.z0,
                    rect.z1,
                    rect.x,
                    rect.positive.unwrap_or(true),
                ),
                projection
            )
        }
        AnyShape::ZXRect(rect) => {
            wrap!(
                hittable::aa_rect::ZXRect::new(
                    rect.z0,
                    rect.z1,
                    rect.x0,
                    rect.x1,
                    rect.y,
                    rect.positive.unwrap_or(true),
                ),
                projection
            )
        }
        AnyShape::ConstantMedium(medium) => {
            let shapes = visit_shape(
                context,
                match &*medium.shape {
                    NameOrShape::Name(name) => context.name_shapes[name],
                    NameOrShape::Shape(shape) => shape,
                },
            );
            if shapes.len() == 1 {
                wrap_no_sample!(
                    hittable::constant_medium::ConstantMedium::new(
                        shapes[0].0.clone(),
                        medium.density,
                    ),
                    projection
                )
            } else {
                wrap_no_sample!(
                    hittable::constant_medium::ConstantMedium::new(
                        shapes
                            .into_iter()
                            .map(|x| x.0.clone())
                            .collect::<hittables::group::HittableGroup<_>>(),
                        medium.density,
                    ),
                    projection
                )
            }
        }
        AnyShape::Triangle(triangle) => {
            let vertices = triangle.vertices.map(|x| Point3::new(x[0], x[1], x[2]));
            let normals = triangle.normals.map_or_else(
                || {
                    let norm = (vertices[1] - vertices[0])
                        .cross(&(vertices[2] - vertices[1]))
                        .normalize();
                    [norm; 3]
                },
                |x| x.map(|y| Vector3::new(y[0], y[1], y[2])),
            );
            let uvs = triangle.uvs.map_or_else(
                || [Vector2::new(0., 0.); 3],
                |x| x.map(|y| Vector2::new(y[0], y[1])),
            );
            wrap!(
                hittable::triangle::Triangle::new(vertices, normals, uvs,),
                projection
            )
        }
        AnyShape::Cuboid(cuboid) => hittables::cuboid::Cuboid::new(
            Point3::new(cuboid.p0[0], cuboid.p0[1], cuboid.p0[2]),
            Point3::new(cuboid.p1[0], cuboid.p1[1], cuboid.p1[2]),
        )
        .into_hittables()
        .map(|(x, y)| {
            if let Some(projection) = projection {
                (
                    Arc::new(hittable::transform::TransformHittable::new(projection, x))
                        as BoxedHittable<F, R>,
                    y.map(|x| {
                        Arc::new(hittable::transform::TransformHittable::new(projection, x))
                            as BoxedSamplable<F, R>
                    }),
                )
            } else {
                (x, y)
            }
        })
        .collect(),
        AnyShape::Mesh(mesh) => {
            let (models, _materials) = tobj::load_obj(
                &mesh.file,
                &tobj::LoadOptions {
                    triangulate: true,
                    ..Default::default()
                },
            )
            .expect("cannot open obj file");
            let mesh = match &mesh.model {
                None => &models[0].mesh,
                Some(IntOrString::Int(i)) => &models[*i].mesh,
                Some(IntOrString::String(name)) => {
                    &models
                        .iter()
                        .find(|m| m.name == *name)
                        .expect("cannot find the model")
                        .mesh
                }
            };
            ManyHittables::<F, R>::into_hittables(hittables::obj::Mesh::new(mesh))
                .map(|(x, y)| {
                    if let Some(projection) = projection {
                        (
                            Arc::new(hittable::transform::TransformHittable::new(projection, x))
                                as BoxedHittable<F, R>,
                            y.map(|x| {
                                Arc::new(hittable::transform::TransformHittable::new(projection, x))
                                    as BoxedSamplable<F, R>
                            }),
                        )
                    } else {
                        (
                            x as BoxedHittable<F, R>,
                            y.map(|x| x as BoxedSamplable<F, R>),
                        )
                    }
                })
                .collect()
        }
    };
    if let Some(name) = shape.name.as_ref() {
        context.visiting_shapes.remove(name);
        context.shapes.insert(name.clone(), result.clone());
    }
    result
}

fn visit_object<F, R: 'static + Rng>(context: &mut VisitContext<F, R>, obj: &Object)
where
    F: SimdF32Field + Into<[f32; F::LANES]> + From<[f32; F::LANES]>,
    F::SimdBool: SimdBoolField<F>,
{
    let material = visit_material(
        context,
        match &obj.material {
            NameOrMaterial::Name(name) => context.name_materials[name],
            NameOrMaterial::Material(mat) => mat,
        },
    );
    let shape = visit_shape(
        context,
        match &obj.shape {
            NameOrShape::Name(name) => context.name_shapes[name],
            NameOrShape::Shape(shape) => shape,
        },
    );
    let visible = obj.visible.unwrap_or(true);
    let important = obj.important.unwrap_or(false);
    if visible && important {
        shape.into_iter().for_each(|(hit, samp)| {
            if let Some(samp) = samp {
                context.scene.add_important(hit, samp, material.clone());
            } else {
                eprintln!("importance sampling on unsupported shape!");
            }
        });
    } else if visible {
        shape.into_iter().for_each(|(hit, _samp)| {
            context.scene.add(hit, material.clone());
        });
    }
}

pub fn build_scene<F, R: 'static + Rng>(param: &SceneParam) -> Scene<F, R>
where
    F: SimdF32Field + Into<[f32; F::LANES]> + From<[f32; F::LANES]>,
    F::SimdBool: SimdBoolField<F>,
{
    let mut scene = Scene::new(
        param.background.map_or(Vector3::from_element(0f32), |x| {
            Vector3::new(x[0], x[1], x[2])
        }),
        param.environment.map_or(Vector3::from_element(0f32), |x| {
            Vector3::new(x[0], x[1], x[2])
        }),
    );
    let mut context = VisitContext::<F, R>::new(&mut scene, param);
    for object in param.objects.iter() {
        visit_object(&mut context, object);
    }
    scene
}
