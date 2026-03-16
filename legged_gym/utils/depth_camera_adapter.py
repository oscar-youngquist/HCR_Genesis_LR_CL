import torch


class DepthCameraAdapter:
    """Normalize depth camera outputs across simulators.

    Returns metric depth in meters for the latest frame, shaped (num_envs, H, W).
    """

    @staticmethod
    def get_depth_meters_latest(cfg, simulator) -> torch.Tensor:
        # Prefer explicit metric buffer if provided by backend.
        depth_meters = getattr(simulator, "_depth_images_meters", None)
        if depth_meters is not None:
            return depth_meters[:, 0]

        depth = simulator.get_depth_images()
        if depth is None:
            raise RuntimeError("Simulator did not provide depth images")

        cam_cfg = cfg.sensor.depth_camera_config
        near_clip = cam_cfg.near_clip
        far_clip = cam_cfg.far_clip

        if depth.ndim == 4:
            depth_latest = depth[:, 0]
        elif depth.ndim == 5 and depth.shape[-1] != 3:
            depth_latest = depth[:, 0, :, :, 0]
        else:
            raise NotImplementedError(f"Unsupported depth tensor shape: {tuple(depth.shape)}")

        # Heuristic: if values are mostly within [-0.5, 0.5], treat as normalized.
        if depth_latest.min() >= -0.51 and depth_latest.max() <= 0.51:
            return (depth_latest + 0.5) * (far_clip - near_clip) + near_clip
        # Otherwise assume already in meters.
        return depth_latest

