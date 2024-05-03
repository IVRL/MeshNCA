import numpy as np
import torch
import kaolin
from kaolin.render.mesh import prepare_vertices, dibr_rasterization, spherical_harmonic_lighting

from utils.mesh import Mesh
from utils.camera import PerspectiveCamera


class Renderer:
    def __init__(self, height=256, width=256, knum=30, sigmainv=7000,
                 ambient_light=0.28, directional_light=1.0, background_color=1.0):
        """
        :param height: Height of the rendered image
        :param width: Width of the rendered image
        :param knum: Number of nearest faces to consider for each pixel
        :param sigmainv: Smoothing factor for the blending faces to make the rendering differentiable
        :param ambient_light: Intensity of the ambient light
        :param directional_light: Intensity of the directional light
        :param background_color: float, 1.0 is white and 0.0 is black
        """
        self.height = height
        self.width = width
        self.knum = knum
        self.sigmainv = sigmainv

        self.ambient_light = ambient_light
        self.directional_light = directional_light

        self.background_color = background_color

    def render(self, mesh: Mesh, camera: PerspectiveCamera, vertex_features: torch.Tensor,

               harc_clamp=False) -> torch.Tensor:
        """
        Render the mesh using the specified camera.
        :param mesh: Mesh object
        :param camera: PerspectiveCamera object
            should have the following attributes: camera.projection_matrix, camera.transform_matrix
            camera.proj_matrix: torch tensor with shape [3, 1]
            camera.transform_matrix: torch tensor with shape [num_views, 4, 3]
        :param vertex_features: torch tensor with shape [batch_size, num_vertices, num_features]

        :param harc_clamp: bool, whether to clamp the rendered image to [0, 1]

        :return: the rendered images from all views [batch_size, num_views, height, width, num_features]
        """
        device = vertex_features.device
        batch_size = vertex_features.shape[0]
        num_views = camera.transform_matrix.shape[0]

        # face_vertices_world [num_views, num_faces, 3, 3] the world coordinates of the vertices for each face
        # face_vertices_image [num_views, num_faces, 3, 2] the projected coordinates of the vertices for each face
        # face_normals [num_views, num_faces, 3] the normals of the faces
        face_vertices_world, face_vertices_image, face_normals = prepare_vertices(mesh.vertices,
                                                                                  mesh.faces,
                                                                                  camera_proj=camera.projection_matrix,
                                                                                  camera_transform=camera.transform_matrix)

        ## Repeat these tensors to match the batch size
        face_vertices_world = face_vertices_world.repeat(batch_size, 1, 1, 1)  # [batch_size*num_views, ...]
        face_vertices_image = face_vertices_image.repeat(batch_size, 1, 1, 1)  # [batch_size, ...]
        face_normals = face_normals.repeat(batch_size, 1, 1)  # [batch_size, ...]

        face_features = mesh.vertex2face_features(vertex_features)  # [batch_size, num_faces, 3, num_features]
        background = torch.ones(face_features.shape[0], face_features.shape[1], 3, 1, device=device)
        face_features = torch.cat([face_features, background], dim=-1)  # [batch_size, num_faces, 3, num_features+1]
        face_features = face_features.repeat_interleave(repeats=num_views, dim=0)  # [batch_size*num_views, ...]

        # image_features [batch_size*num_views, height, width, num_features + 1] the rendered features for each pixel
        # soft_mask [batch_size*num_views, height, width] the mask showing if a pixel is on the mesh or not
        # face_ids [batch_size*num_views, height, width] the face id for each pixel
        image_features, soft_mask, face_ids = dibr_rasterization(height=self.height, width=self.width,
                                                                 face_vertices_z=face_vertices_world[..., -1],
                                                                 face_vertices_image=face_vertices_image,
                                                                 face_features=face_features,
                                                                 # face_normals_z=face_normals[..., -1],
                                                                 face_normals_z=torch.ones_like(face_normals[..., -1]),
                                                                 knum=self.knum, sigmainv=self.sigmainv,
                                                                 rast_backend='cuda')

        # Using the actual face normals face_normals_z=face_normals[..., -1] will cause artifacts in the rendered image.
        # Instead, we use a constant value of 1.0 for the face normals.

        background = (image_features[..., -1:] < 0.95).float() * self.background_color

        image_features = image_features[..., :-1]
        lighting = self.ambient_light
        if self.directional_light > 0.0:
            with torch.no_grad():
                image_normals = face_normals[
                    torch.arange(batch_size * num_views)[..., None, None, None],
                    face_ids[..., None],
                    torch.arange(3)[None, None, None, :],
                ]  # [batch_size, height, width, 3]
                sh_lights = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0)
                directional_light = spherical_harmonic_lighting(image_normals, sh_lights)  # [batch_size, height, width]
                directional_light = torch.clamp(directional_light, 0.0, 1.0)  # [batch_size, height, width, 1]
                lighting += directional_light * self.directional_light
                lighting = lighting.unsqueeze(-1)

        image_features = image_features * lighting

        image_features = image_features + background
        if harc_clamp:
            image_features = torch.clamp(image_features, 0.0, 1.0)

        image_features = image_features.view(batch_size, num_views, self.height, self.width, -1)
        return image_features



    def __repr__(self):
        return f"Renderer(height={self.height}, width={self.width}, knum={self.knum}, sigmainv={self.sigmainv}, " \
               f"\n\tambient_light={self.ambient_light}, directional_light={self.directional_light}, " \
               f"\n\tbackground_color={self.background_color})"

if __name__ == '__main__':
    from utils.mesh import Mesh
    from utils.camera import PerspectiveCamera

    device = torch.device("cuda:0")

    mesh = Mesh.load_from_obj('../data/meshes/mug/mug.obj', device=device)
    # mesh = Mesh('../data/meshes/flatland/flatland_remesh_lvl1.obj', device=device)
    # mesh = Mesh('../data/meshes/sphere/sphere_remesh_lvl1.obj', device=device)

    with torch.no_grad():
        from utils.video import VideoWriter
        from tqdm import tqdm
        from PIL import Image

        # torch.manual_seed(42)
        np.random.seed(42)
        # Render the mesh from 6 random viewpoints
        camera = PerspectiveCamera.generate_random_view_cameras(6, distance=2.0, device=device)
        renderer = Renderer(height=512, width=512, background_color=1.0)

        vertex_features = torch.zeros((2, mesh.vertices.shape[0], 3), device=device) + 0.5
        vertex_features[1] *= 1.5
        image = renderer.render(mesh, camera, vertex_features).cpu().numpy()
        # image: [batch_size, num_views, height, width, num_features]
        print(image.shape)

        image1 = np.hstack(image[0]) * 255.0
        image2 = np.hstack(image[1]) * 255.0
        image = np.vstack([image1, image2])
        image = np.clip(image, 0, 255).astype(np.uint8)
        Image.fromarray(image).show()

        # with VideoWriter('tmp.mp4', fps=30.0) as video:
        #     for i in tqdm(range(120)):
        #         image = renderer.render(mesh, camera, vertex_features, True).cpu().numpy()
        #         camera.rotateZ(3.0)
        #         image = np.hstack(image)
        #         video.add(image)
        #
