import torch
import torch.nn as nn

from .layers.drop_path import DropPath


class GatedNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        normed = x.permute(0, 2, 3, 1)
        normed = self.norm(normed)
        normed = normed.permute(0, 3, 1, 2).contiguous()
        return self.conv(normed) * x


class Block(nn.Module):
    def __init__(self, dim: int, drop_path: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            GatedNorm(dim),
            nn.Conv2d(dim, 4 * dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * dim, dim, kernel_size=1),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.block(x))


class DepthEncoder(nn.Module):
    DIMS = (128, 256, 512, 1024)
    DEPTHS = (3, 3, 27, 3)

    def __init__(
        self,
        in_chans: int = 6,
        dims: tuple[int, int, int, int] = DIMS,
        depths: tuple[int, int, int, int] = DEPTHS,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.prompt_dims = list(dims)
        all_dims = [dims[0] // 4, dims[0] // 2, *dims]

        drop_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, all_dims[0], kernel_size=3, padding=1),
            self._make_downsample(all_dims[0], all_dims[1]),
        )

        cursor = 0
        self.layer1 = self._make_layer(all_dims[1], all_dims[2], drop_rates[cursor:cursor + depths[0]])
        cursor += depths[0]
        self.layer2 = self._make_layer(all_dims[2], all_dims[3], drop_rates[cursor:cursor + depths[1]])
        cursor += depths[1]
        self.layer3 = self._make_layer(all_dims[3], all_dims[4], drop_rates[cursor:cursor + depths[2]])
        cursor += depths[2]
        self.layer4 = self._make_layer(all_dims[4], all_dims[5], drop_rates[cursor:cursor + depths[3]])

        self.apply(self._init_layers)

    @staticmethod
    def _make_downsample(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            GatedNorm(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def _make_layer(self, in_channels: int, out_channels: int, drop_rates: list[float]) -> nn.Sequential:
        return nn.Sequential(
            self._make_downsample(in_channels, out_channels),
            *[Block(out_channels, drop_rate) for drop_rate in drop_rates],
        )

    @staticmethod
    def _init_layers(module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)
        return [p1, p2, p3, p4]
