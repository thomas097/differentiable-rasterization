import torch
from typing import Iterable
from camera_models import CameraModel


class PerspectiveCamera(torch.nn.Module):
    def __init__(
            self, 
            location: Iterable, 
            rotation: Iterable, 
            camera_model: CameraModel, 
            trainable: bool = True
            ) -> None:
        """Pytorch implementation of a perspective camera.

        Args:
            location (Iterable):        Location of camera origin in world of the form (x, y, z).
            rotation (Iterable):        Orientation in quaternions (qw, qx, qy, qz).
            camera_model (CameraModel): Camera model mapping vertices in camera space to pixel coordinates.
            trainable (bool, optional): Whether to optimize the intrinsics (Default: True).
        """
        super().__init__()
        tvec = torch.tensor(location, dtype=torch.float32).unsqueeze(0)
        qvec = torch.tensor(rotation, dtype=torch.float32)

        if trainable:
            self.tvec = torch.nn.Parameter(tvec, requires_grad=True)
            self.qvec = torch.nn.Parameter(qvec, requires_grad=True)
        else:
            self.register_buffer('tvec', location, persistent=True)
            self.register_buffer('qvec', rotation, persistent=True)

        self.camera_model = camera_model

    def get_3x3_rotation_matrix(self) -> torch.Tensor:
        """Converts a 4D vector of quaternions (qw, qx, qy, qz) to a 3x3 rotation matrix.

        Returns:
            torch.Tensor: Rotation matrix of shape (3, 3).
        """
        qw, qx, qy, qz = self.qvec

        r00 = 1 - 2 * qy ** 2 - 2 * qz ** 2
        r01 = 2 * qx * qy - 2 * qw * qz
        r02 = 2 * qz * qx + 2 * qw * qy

        r10 = 2 * qx * qy + 2 * qw * qz
        r11 = 1 - 2 * qx ** 2 - 2 * qz ** 2
        r12 = 2 * qy * qz - 2 * qw * qx

        r20 = 2 * qz * qx - 2 * qw * qy
        r21 = 2 * qy * qz + 2 * qw * qx
        r22 = 1 - 2 * qx ** 2 - 2 * qy ** 2

        rot = torch.stack([
            torch.stack([r00, r01, r02]),
            torch.stack([r10, r11, r12]),
            torch.stack([r20, r21, r22]),
        ])
        return rot
    
    def world_to_cam(self, verts: torch.Tensor) -> torch.Tensor:
        """Projects vertices in world coordinates to camera coordinates.

        Args:
            verts (torch.Tensor): Matrix of vertices with shape (N, 3).

        Returns:
            torch.Tensor: Matrix of vertices with shape (N, 3).
        """
        rot = self.get_3x3_rotation_matrix()
        return torch.matmul(verts, rot.t()) + self.tvec 
    
    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        """Projects vertices in world coordinates to image space.

        Args:
            verts (torch.Tensor): Matrix of vertices with shape (N, 3).

        Returns:
            torch.Tensor: Matrix of image coordinates (N, 3).
        """
        uvw = self.world_to_cam(verts)
        return self.camera_model(uvw)


    

if __name__ == '__main__':
    from camera_models import OpenCVCameraModel

    camera = PerspectiveCamera(
        location=(3.0, 1.0, 2.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
        camera_model=OpenCVCameraModel(params=(256, 256, 128, 128, 0.3, 0.2, 0.01, 0.02))
    )
    print(camera(torch.rand(5, 3)))