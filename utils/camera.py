import numpy as np
import torch
import kaolin
from kaolin.render.camera import generate_transformation_matrix, generate_perspective_projection


class PerspectiveCamera:
    """
    This class represents a batch of cameras in 3D space.
    """

    def __init__(self, fov=60.0,
                 elevation=[0.0], azimuth=[0.0], distance=[2.0],
                 look_at=[0.0, 0.0, 0.0], up_vector=[0.0, 1.0, 0.0], device='cuda:0'):
        """
        :param elevation: Elevation angles of the cameras in degrees (list or numpy array)
        :param azimuth: Azimuth angles of the cameras in degrees (list or numpy array)
        :param distance: Distances of the cameras from the origin (list or numpy array)
        :param look_at: Point the camera is looking at (shared by all cameras)
        :param up_vector: Up vector of the camera (shared by all cameras)
        :param fov: Field of view of the camera in degrees (shared by all cameras)
        :param device: PyTorch device to store the camera data

        The camera class has the following attributes:
        transform_matrix: torch tensor with shape [num_cameras, 4, 3]
        projection_matrix: torch tensor with shape [3, 1]
        position: torch tensor with shape [num_cameras, 3]
        """

        with torch.no_grad():
            self.elevation = torch.tensor(elevation, dtype=torch.float32, device=device) * torch.pi / 180.0
            self.azimuth = torch.tensor(azimuth, dtype=torch.float32, device=device) * torch.pi / 180.0
            self.fov = fov * torch.pi / 180.0
            self.distance = torch.tensor(distance, dtype=torch.float32, device=device)
            self.look_at = torch.tensor(look_at, dtype=torch.float32, device=device)
            self.up_vector = torch.tensor(up_vector, dtype=torch.float32, device=device)

            self._update_camera()

    def _update_camera(self):
        x = self.distance * torch.cos(self.elevation) * torch.cos(self.azimuth)
        y = self.distance * torch.sin(self.elevation)
        z = self.distance * torch.cos(self.elevation) * torch.sin(self.azimuth)
        self.position = torch.stack([x, y, z], dim=1)

        self.transform_matrix = generate_transformation_matrix(self.position, self.look_at.unsqueeze(0),
                                                               self.up_vector.unsqueeze(0)).to(self.azimuth.device)

        self.projection_matrix = generate_perspective_projection(self.fov, dtype=torch.float32).to(self.azimuth.device)

    def rotateY(self, angle):
        self.azimuth += angle * torch.pi / 180.0
        self._update_camera()

    @staticmethod
    def generate_random_view_cameras(num_views, distance=2.5, **kwargs):
        azimuth = np.random.rand(num_views) * 360.0
        elevation = np.arcsin(np.random.rand(num_views) * 2.0 - 1.0) * 180.0 / np.pi
        distance = np.ones(num_views) * distance

        return PerspectiveCamera(elevation=elevation, azimuth=azimuth, distance=distance, **kwargs)

    def __repr__(self):
        return f"Camera(" \
               f"\n\tNumber of Cameras = {self.transform_matrix.shape[0]}," \
               f"\n\tElevation = {self.elevation * 180.0 / torch.pi}," \
               f"\n\tAzimuth = {self.azimuth * 180.0 / torch.pi}," \
               f"\n\tDistance = {self.distance}," \
               f"\n\tField of View = {self.fov * 180.0 / torch.pi}," \
               f"\n\tLook At = {self.look_at}," \
               f"\n\tUp Vector = {self.up_vector}," \
               f"\n)"


if __name__ == '__main__':
    cam = PerspectiveCamera.generate_random_view_cameras(3, distance=2.0)
    print(cam)
