import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import create_uv_grid, position_grid_to_embed


def _make_scratch(in_shape, out_shape, groups=1):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class ResidualConvBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.activation = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features,
        prompt_dim=None,
        align_corners=True,
        size=None,
    ):
        super().__init__()

        self.align_corners = align_corners

        self.out_conv = nn.Conv2d(
            features, features, kernel_size=1, stride=1, padding=0, bias=True, groups=1
        )
        self.resConfUnit1 = ResidualConvBlock(features)
        self.resConfUnit2 = ResidualConvBlock(features)

        if prompt_dim:
            self.resConfUnit_prompt = nn.Sequential(
                nn.Conv2d(prompt_dim, features, kernel_size=3, stride=1, padding=1, bias=True, groups=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=1)
            )

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, prompt=None, size=None):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if prompt is not None and hasattr(self, "resConfUnit_prompt"):
            prompt = F.interpolate(prompt, output.shape[2:], mode="bilinear", align_corners=False)
            res = self.resConfUnit_prompt(prompt)
            output = self.skip_add.add(output, res)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = F.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)
        return output


def _make_fusion_block(features, size=None, **kwargs):
    return FeatureFusionBlock(
        features,
        align_corners=True,
        size=size,
        **kwargs,
    )


class Head(nn.Module):
    def __init__(
        self,
        in_channels: int,
        pos_embed: bool = False,
        features: int = 256,
        out_channels: list[int] | int = [256, 512, 1024, 2048],
        depth_prompt_dim: list[int] | int | None = None,
        ray_output_dim: int = 2,
        depth_output_dim: int = 1,
    ):
        super().__init__()
        self.pos_embed = pos_embed

        if isinstance(out_channels, int):
            out_channels = [out_channels] * 4
        self.depth_prompt_dim = self._prepare_depth_prompt(depth_prompt_dim)

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            self._make_resampler(out_channel, out_channel, scale_factor)
            for out_channel, scale_factor in zip(out_channels, (4, 2, 1, 0.5))
        ])

        self.scratch = _make_scratch(out_channels, features, groups=1)

        self.scratch.refinenet1_depth = _make_fusion_block(features, prompt_dim=self.depth_prompt_dim[0])
        self.scratch.refinenet2_depth = _make_fusion_block(features, prompt_dim=self.depth_prompt_dim[1])
        self.scratch.refinenet3_depth = _make_fusion_block(features, prompt_dim=self.depth_prompt_dim[2])
        self.scratch.refinenet4_depth = _make_fusion_block(features, prompt_dim=self.depth_prompt_dim[3])

        self.scratch.refinenet1_ray = _make_fusion_block(features)
        self.scratch.refinenet2_ray = _make_fusion_block(features)
        self.scratch.refinenet3_ray = _make_fusion_block(features)
        self.scratch.refinenet4_ray = _make_fusion_block(features)

        self.scratch.output_conv1_depth = self._make_out1_block(features)
        self.scratch.output_conv2_depth = self._make_out2_block(features, depth_output_dim)

        self.scratch.output_conv1_ray = self._make_out1_block(features)
        self.scratch.output_conv2_ray = self._make_out2_block(features, ray_output_dim)

    @staticmethod
    def _make_out1_block(features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        )

    @staticmethod
    def _make_out2_block(features: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_dim, kernel_size=1, stride=1, padding=0),
        )

    def _make_resampler(self, in_channels, out_channels, scale_factor):
        if scale_factor > 1:
            resampler = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=int(scale_factor),
                    stride=int(scale_factor),
                    padding=0,
                ),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
            resampler[0].weight.data[:] = resampler[0].weight.data[:, :, :1, :1]
        elif scale_factor in [1, 0.5]:
            stride = int(1 // scale_factor)
            resampler = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            )
        else:
            raise NotImplementedError(f"Unsupported scale factor: {scale_factor}")
        return resampler

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def _project_features(self, features, out_h, out_w):
        out = []
        for i, x in enumerate(features):
            x = self.projects[i](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, out_w, out_h)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        return (
            self.scratch.layer1_rn(layer_1),
            self.scratch.layer2_rn(layer_2),
            self.scratch.layer3_rn(layer_3),
            self.scratch.layer4_rn(layer_4),
        )

    def _prepare_depth_prompt(self, depth_prompt):
        if not isinstance(depth_prompt, (list, tuple)):
            depth_prompt = [depth_prompt] * 4
        return depth_prompt

    def _predict(self, x, output_conv2, out_h, out_w):
        x = F.interpolate(x, (out_h, out_w), mode="bilinear", align_corners=True)
        if self.pos_embed:
            x = self._apply_pos_embed(x, out_w, out_h)
        return output_conv2(x)

    def forward(self, features, out_h, out_w, depth_prompt=None, chunk_size=8):
        batch_size = features[0].shape[0]
        depth_prompt = self._prepare_depth_prompt(depth_prompt)

        if chunk_size is None or chunk_size >= batch_size:
            return self._forward_impl(features, out_h, out_w, depth_prompt)

        log_depths = []
        rays = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            feature_chunk = [x[start:end] for x in features]
            prompt_chunk = [x[start:end] if x is not None else None for x in depth_prompt]
            log_depth, ray = self._forward_impl(feature_chunk, out_h, out_w, prompt_chunk)
            log_depths.append(log_depth)
            rays.append(ray)

        return torch.cat(log_depths, dim=0), torch.cat(rays, dim=0)

    def _forward_impl(self, features, out_h, out_w, depth_prompt):
        layer_1, layer_2, layer_3, layer_4 = self._project_features(features, out_h, out_w)

        depth = self.scratch.refinenet4_depth(
            layer_4, size=layer_3.shape[2:], prompt=depth_prompt[3]
        )
        ray = self.scratch.refinenet4_ray(layer_4, size=layer_3.shape[2:])

        depth = self.scratch.refinenet3_depth(
            depth, layer_3, size=layer_2.shape[2:], prompt=depth_prompt[2]
        )
        ray = self.scratch.refinenet3_ray(ray, layer_3, size=layer_2.shape[2:])

        depth = self.scratch.refinenet2_depth(
            depth, layer_2, size=layer_1.shape[2:], prompt=depth_prompt[1]
        )
        ray = self.scratch.refinenet2_ray(ray, layer_2, size=layer_1.shape[2:])

        depth = self.scratch.refinenet1_depth(depth, layer_1, prompt=depth_prompt[0])
        ray = self.scratch.refinenet1_ray(ray, layer_1)

        depth = self.scratch.output_conv1_depth(depth)
        ray = self.scratch.output_conv1_ray(ray)

        log_depth = self._predict(depth, self.scratch.output_conv2_depth, out_h, out_w)
        ray = self._predict(ray, self.scratch.output_conv2_ray, out_h, out_w)
        return log_depth, ray
