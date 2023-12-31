import torch
from typing import Iterable


class CameraModel(torch.nn.Module):
    def __init__(self, params: Iterable, trainable: bool = True) -> None:
        """Pytorch implementation of a pinhole camera model to project 
        vertices in camera coordinates to image coordinates.

        Args:
            params (Iterable): Camera intrinsics (fx, fy, cx, cy).
            trainable (bool, optional): Whether to optimize the intrinsics (Default: True).
        """
        super().__init__()
        params = torch.tensor(params, dtype=torch.float32)
        
        if trainable:
            self.data = torch.nn.Parameter(params, requires_grad=True)
        else:
            self.register_buffer('data', params, persistent=True)

    def _distortion(self, u: torch.Tensor, v: torch.Tensor, extra_params: torch.Tensor) -> tuple:
        return 0.0, 0.0

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        """Projects vertices in camera coordinates to image space.

        Args:
            verts (torch.Tensor): Matrix of vertices with shape (N, 3).

        Returns:
            torch.Tensor: Matrix of image coordinates (N, 3).
        """
        # Perspective transform
        x, y, z = torch.split(verts, 1, dim=1)
        u = x / z
        v = y / z

        # Radial / tangential distortion
        du, dv = self._distortion(u, v, extra_params=self.data[4:])

        # Image coordinates
        fx, fy, cx, cy = self.data[:4]
        u = fx * (u + du) + cx
        v = fy * (v + dv) + cy
        return torch.concat([u, v], dim=1)
    

class OpenCVCameraModel(CameraModel):
    def __init__(self, params: Iterable, trainable: bool = True) -> None:
        """Pytorch implementation of OpenCV camera model with 
        distortion parameters: fx, fy, cx, cy, k1, k2, p1, p2
        
        For details, see:
        https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
        """
        super().__init__(params=params, trainable=trainable)

    def _distortion(self, x: torch.Tensor, y: torch.Tensor, extra_params: torch.Tensor) -> tuple:
        k1, k2, p1, p2 = extra_params
        x2 = x * x
        xy = x * y
        y2 = y * y
        r2 = x2 + y2
        radial = k1 * r2 + k2 * r2 * r2
        dx = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
        dy = y * radial + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)
        return dx, dy
    

if __name__ == '__main__':
    model = OpenCVCameraModel(params=(256, 256, 128, 128, 0.3, 0.2, 0.01, 0.02))
    uv = model(verts=torch.rand(5, 3))
    print(uv)