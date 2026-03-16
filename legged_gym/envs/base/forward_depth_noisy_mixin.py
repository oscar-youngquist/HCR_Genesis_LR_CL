import random
from typing import List

import torch
import torch.nn.functional as F


class ForwardDepthNoisyMixin:
    """Parkour-style depth processing + latency for `forward_depth`.

    Expects `self.sensor_tensor_dict["forward_depth"]` to be a list of per-env (H, W) depth images
    in meters (positive).
    """

    def _init_buffers(self):
        super()._init_buffers()
        if not getattr(self.cfg.sensor, "add_depth", False):
            return

        cam_cfg = self.cfg.sensor.depth_camera_config
        # latency buffer in seconds (per env)
        self.forward_camera_latency_buffer = torch.zeros(
            (self.num_envs,),
            dtype=torch.float32,
            device=self.device,
        )
        latency_range = getattr(cam_cfg, "latency_range", (0.0, 0.0))
        if latency_range[1] > 0:
            self.forward_camera_latency_buffer[:] = (
                (latency_range[1] - latency_range[0]) * torch.rand_like(self.forward_camera_latency_buffer)
                + latency_range[0]
            )

        self.forward_camera_delayed_frames = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )

        # obs buffer stores processed frames (buffer_length, N, 1, H_out, W_out)
        buffer_length = int(latency_range[1] / self.dt) + 1
        H_out, W_out = getattr(cam_cfg, "output_resolution", (cam_cfg.resolution[1], cam_cfg.resolution[0]))
        self.forward_depth_obs_buffer = torch.zeros(
            (buffer_length, self.num_envs, 1, H_out, W_out),
            dtype=torch.float32,
            device=self.device,
        )
        self.forward_depth_obs_refreshed = False
        self.forward_depth_output = torch.zeros(
            (self.num_envs, 1, H_out, W_out),
            dtype=torch.float32,
            device=self.device,
        )

        # kernels for contour detection (copied conceptually from parkour)
        self.contour_detection_kernel = torch.zeros((8, 1, 3, 3), dtype=torch.float32, device=self.device)
        self.contour_detection_kernel[0, :, 1, 1] = 0.5
        self.contour_detection_kernel[0, :, 0, 0] = -0.5
        self.contour_detection_kernel[1, :, 1, 1] = 0.1
        self.contour_detection_kernel[1, :, 0, 1] = -0.1
        self.contour_detection_kernel[2, :, 1, 1] = 0.5
        self.contour_detection_kernel[2, :, 0, 2] = -0.5
        self.contour_detection_kernel[3, :, 1, 1] = 1.2
        self.contour_detection_kernel[3, :, 1, 0] = -1.2
        self.contour_detection_kernel[4, :, 1, 1] = 1.2
        self.contour_detection_kernel[4, :, 1, 2] = -1.2
        self.contour_detection_kernel[5, :, 1, 1] = 0.5
        self.contour_detection_kernel[5, :, 2, 0] = -0.5
        self.contour_detection_kernel[6, :, 1, 1] = 0.1
        self.contour_detection_kernel[6, :, 2, 1] = -0.1
        self.contour_detection_kernel[7, :, 1, 1] = 0.5
        self.contour_detection_kernel[7, :, 2, 2] = -0.5

    def set_buffers_refreshed_to_false(self):
        if hasattr(self, "forward_depth_obs_refreshed"):
            self.forward_depth_obs_refreshed = False

    def _add_depth_contour(self, depth_images: torch.Tensor) -> torch.Tensor:
        cam_cfg = self.cfg.sensor.depth_camera_config
        noise_cfg = getattr(self.cfg.noise, "forward_depth", None)
        if noise_cfg is None:
            return depth_images
        k = getattr(noise_cfg, "contour_detection_kernel_size", 3)
        thr = getattr(noise_cfg, "contour_threshold", 0.0)
        if thr <= 0.0:
            return depth_images
        mask = (
            F.max_pool2d(
                torch.abs(F.conv2d(depth_images, self.contour_detection_kernel, padding=1))
                .max(dim=-3, keepdim=True)[0],
                kernel_size=k,
                stride=1,
                padding=int(k / 2),
            )
            > thr
        )
        depth_images = depth_images.clone()
        depth_images[mask] = 0.0
        return depth_images

    def _add_depth_random_holes(self, depth_images: torch.Tensor) -> torch.Tensor:
        """Cheap proxy for stereo artifacts: randomly zero out pixels."""
        noise_cfg = getattr(self.cfg.noise, "forward_depth", None)
        if noise_cfg is None:
            return depth_images
        p = getattr(noise_cfg, "artifacts_prob", 0.0)
        if p <= 0.0:
            return depth_images
        N, _, H, W = depth_images.shape
        mask = (torch.rand((N, 1, H, W), device=depth_images.device) < p) & (depth_images > 0.0)
        depth_images = depth_images.clone()
        depth_images[mask] = 0.0
        return depth_images

    def _add_depth_stereo_noise(self, depth_images: torch.Tensor) -> torch.Tensor:
        """Distance-dependent noise model (far noisier than near)."""
        noise_cfg = getattr(self.cfg.noise, "forward_depth", None)
        if noise_cfg is None:
            return depth_images
        far_d = getattr(noise_cfg, "stereo_far_distance", 0.0)
        far_std = getattr(noise_cfg, "stereo_far_noise_std", 0.0)
        near_std = getattr(noise_cfg, "stereo_near_noise_std", 0.0)
        if far_d <= 0.0 or (far_std <= 0.0 and near_std <= 0.0):
            return depth_images
        far_mask = depth_images > far_d
        near_mask = ~far_mask
        depth_images = depth_images.clone()
        if far_std > 0.0:
            depth_images[far_mask] += torch.randn_like(depth_images[far_mask]) * far_std
        if near_std > 0.0:
            depth_images[near_mask] += torch.randn_like(depth_images[near_mask]) * near_std
        return depth_images

    @torch.no_grad()
    def _process_depth_image(self, depth_images: List[torch.Tensor]) -> torch.Tensor:
        """Returns processed depth frames shaped (1, N, 1, H_out, W_out), values in [0, 1]."""
        cam_cfg = self.cfg.sensor.depth_camera_config
        depth = torch.stack(depth_images).unsqueeze(1).contiguous()  # (N, 1, H, W) meters

        # optional artifacts
        depth = self._add_depth_contour(depth)
        depth = self._add_depth_stereo_noise(depth)
        depth = self._add_depth_random_holes(depth)

        # normalize to [0,1] using depth_range (meters)
        depth_range = getattr(cam_cfg, "depth_range", (cam_cfg.near_clip, cam_cfg.far_clip))
        depth = torch.clamp(depth, depth_range[0], depth_range[1])
        depth = (depth - depth_range[0]) / (depth_range[1] - depth_range[0])

        # crop
        crop_tb = getattr(cam_cfg, "crop_top_bottom", (0, 0))
        crop_lr = getattr(cam_cfg, "crop_left_right", (0, 0))
        if crop_tb != (0, 0) or crop_lr != (0, 0):
            H, W = depth.shape[-2:]
            depth = depth[
                ...,
                crop_tb[0] : H - crop_tb[1],
                crop_lr[0] : W - crop_lr[1],
            ]

        # resize to output_resolution if provided
        out_res = getattr(cam_cfg, "output_resolution", None)  # (H_out, W_out)
        if out_res is not None:
            depth = F.interpolate(depth, size=out_res, mode="bicubic", align_corners=False)

        depth = depth.clamp(0.0, 1.0)
        return depth.unsqueeze(0)  # (1, N, 1, H_out, W_out)

    @torch.no_grad()
    def _get_forward_depth_obs(self) -> torch.Tensor:
        """Returns flattened processed+delayed depth: (num_envs, H_out*W_out)."""
        if not getattr(self.cfg.sensor, "add_depth", False):
            return super()._get_forward_depth_obs()

        if hasattr(self, "forward_depth_obs_buffer") and not self.forward_depth_obs_refreshed:
            # append latest processed frame
            self.forward_depth_obs_buffer = torch.cat(
                [
                    self.forward_depth_obs_buffer[1:],
                    self._process_depth_image(self.sensor_tensor_dict["forward_depth"]),
                ],
                dim=0,
            )

            cam_cfg = self.cfg.sensor.depth_camera_config
            refresh_duration = getattr(cam_cfg, "refresh_duration", self.dt)
            if refresh_duration <= 0.0:
                refresh_duration = self.dt
            delay_refresh_mask = (self.episode_length_buf % int(refresh_duration / self.dt) == 0)

            frame_select = (self.forward_camera_latency_buffer / self.dt).to(int)
            self.forward_camera_delayed_frames = torch.where(
                delay_refresh_mask,
                torch.minimum(frame_select, self.forward_camera_delayed_frames + 1),
                self.forward_camera_delayed_frames + 1,
            )
            self.forward_camera_delayed_frames = torch.clamp(
                self.forward_camera_delayed_frames, 0, self.forward_depth_obs_buffer.shape[0]
            )
            self.forward_depth_output = self.forward_depth_obs_buffer[
                -self.forward_camera_delayed_frames,
                torch.arange(self.num_envs, device=self.device),
            ].clone()
            self.forward_depth_obs_refreshed = True

        return self.forward_depth_output.flatten(start_dim=1)

