from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_encoder import DepthEncoder
from .head import Head
from .layers import DINOv2, model_configs
from .poisson_completion import poisson_completion
from .utils import log, median


class LDCMModel(nn.Module):
    PATCH_SIZE = 14

    def __init__(self, encoder: str = "vitb"):
        super().__init__()

        self.moge = None

        self.pretrained = DINOv2(model_name=encoder)
        self.depth_encoder = DepthEncoder(in_chans=6)

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        model_cfg = model_configs[encoder]
        self.layer_idxs = model_cfg["layer_idxs"]
        self.head = Head(
            in_channels=self.pretrained.embed_dim,
            features=model_cfg["features"],
            out_channels=model_cfg["out_channels"],
            depth_prompt_dim=self.depth_encoder.prompt_dims,
            ray_output_dim=2,
            depth_output_dim=1,
            pos_embed=True,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "pkqbajng/LDCM",
        moge_path: str = "Ruicheng/moge-2-vits-normal",
        map_location: str | torch.device = "cpu",
        **hf_kwargs,
    ):
        from huggingface_hub import hf_hub_download

        from .moge.model.v2 import MoGeModel

        path = Path(pretrained_model_name_or_path).expanduser()
        if path.exists():
            checkpoint_path = path / "ldcm.pt" if path.is_dir() else path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="ldcm.pt",
                **hf_kwargs,
            )

        model = cls()
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("moge.")
        }
        model.load_state_dict(state_dict, strict=True)

        model.moge = MoGeModel.from_pretrained(moge_path)
        model.moge.eval()
        for param in model.moge.parameters():
            param.requires_grad = False
        return model

    @staticmethod
    def _valid_depth_mask(depth: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(depth) & (depth > 0)

    @staticmethod
    def _masked_log_depth(depth: torch.Tensor, mask: torch.Tensor, medians: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, log(depth / medians), torch.zeros_like(depth))

    @staticmethod
    def _resize_to_multiple(value: int, multiple: int) -> int:
        return max(multiple, int(round(value / multiple)) * multiple)

    @staticmethod
    def _resize_nchw(tensor: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
        if tensor.shape[-2:] == size:
            return tensor
        if mode == "nearest":
            return F.interpolate(tensor, size=size, mode=mode)
        return F.interpolate(tensor, size=size, mode=mode, align_corners=False)

    @classmethod
    def _resize_output(cls, output: dict, size: tuple[int, int]) -> dict:
        resized = dict(output)
        for key in ("mono_depth", "coarse_depth", "depth_pred", "points_pred"):
            if key in resized:
                resized[key] = cls._resize_nchw(resized[key], size, mode="bilinear")
        if "mask" in resized:
            mask = cls._resize_nchw(resized["mask"].float(), size, mode="nearest")
            resized["mask"] = mask.bool()
        return resized

    @staticmethod
    def _format_output(output: dict) -> dict:
        output = dict(output)
        output["points_pred"] = output["points_pred"].permute(0, 2, 3, 1)
        return output

    @torch.no_grad()
    def prepare_input(self, image: torch.Tensor, prior: torch.Tensor):
        if self.moge is None:
            raise RuntimeError("MoGe is not loaded. Use LDCMModel.from_pretrained(...).")
        moge_result = self.moge.infer(image.half())
        mono_depth = moge_result["depth"].unsqueeze(1).float()
        mono_mask = moge_result["mask"].unsqueeze(1).bool()
        mono_depth = torch.where(mono_mask, mono_depth, torch.zeros_like(mono_depth))

        device_type = image.device.type
        with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=device_type == "cuda"):
            prior = prior.float()
            sparse_mask = self._valid_depth_mask(prior)
            sparse = torch.where(sparse_mask, prior, torch.zeros_like(prior))
            medians = median(sparse)

            coarse_depth = poisson_completion(
                sparse=sparse,
                mono_depth=mono_depth,
                num_scales=3,
                thres=3.0,
                lamda=5.0,
                rtol=1e-5,
                max_iter_per_scale=[2000, 1000, 500],
                confidence=mono_mask.float(),
                max_resolution_ratio=0.25,
            )

            coarse_mask = mono_mask & self._valid_depth_mask(coarse_depth)
            coarse_depth = torch.where(coarse_mask, coarse_depth, torch.zeros_like(coarse_depth))

            sparse_log = self._masked_log_depth(sparse, sparse_mask, medians)
            poisson_log = self._masked_log_depth(coarse_depth, coarse_mask, medians)
            encoder_input = torch.cat([sparse_log, sparse_mask.to(sparse_log.dtype), poisson_log], dim=1)

        return encoder_input, medians, mono_depth, mono_mask, coarse_depth

    def _forward_nchw(self, image: torch.Tensor, prior: torch.Tensor):
        img_h, img_w = image.shape[-2:]
        encoder_input, medians, mono_depth, mono_mask, coarse_depth = self.prepare_input(image, prior)

        x = (image - self.image_mean) / self.image_std
        device_type = image.device.type

        with torch.autocast(device_type=device_type, enabled=device_type == "cuda"):
            features = self.pretrained.get_intermediate_layers(
                x,
                self.layer_idxs,
                reshape=True,
                return_class_token=False,
            )
            depth_prompt = self.depth_encoder(torch.cat([x, encoder_input], dim=1))

        with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=device_type == "cuda"):
            log_depth, ray_pred = self.head(features, out_h=img_h, out_w=img_w, depth_prompt=depth_prompt)
            depth_pred = log_depth.clamp(min=-10, max=10).exp() * medians
            points_pred = torch.cat([ray_pred * depth_pred, depth_pred], dim=1)

        return {
            "mono_depth": mono_depth,
            "coarse_depth": coarse_depth,
            "depth_pred": depth_pred.clamp(min=1e-4),
            "points_pred": points_pred,
            "mask": mono_mask,
        }

    def forward(self, image: torch.Tensor, prior: torch.Tensor):
        return self._format_output(self._forward_nchw(image, prior))

    @torch.no_grad()
    def infer(self, image: torch.Tensor, prior: torch.Tensor):
        raw_h, raw_w = image.shape[-2:]
        tgt_h = self._resize_to_multiple(raw_h, self.PATCH_SIZE)
        tgt_w = self._resize_to_multiple(raw_w, self.PATCH_SIZE)
        target_size = (tgt_h, tgt_w)

        image = self._resize_nchw(image, target_size, mode="bilinear")
        prior = self._resize_nchw(prior, target_size, mode="nearest")
        output = self._forward_nchw(image, prior)
        output = self._resize_output(output, (raw_h, raw_w))
        return self._format_output(output)
