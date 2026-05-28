from pathlib import Path
import argparse
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ldcm import LDCMModel


matplotlib.use("Agg")
COLORMAP = matplotlib.colormaps["jet"]


def load_image(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


def load_depth(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        depth = np.load(path).astype(np.float32)
    else:
        depth = np.asarray(Image.open(path), dtype=np.float32) / 1000.0
    return torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)


def save_depth_uint16(depth_m: np.ndarray, path: Path) -> None:
    depth_mm = np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(depth_mm).save(path)


def colorize_depth(depth: np.ndarray, valid: np.ndarray | None = None, value_range=None) -> Image.Image:
    if valid is None:
        valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return Image.fromarray(np.zeros((*depth.shape, 3), dtype=np.uint8))
    if value_range is None:
        values = depth[valid]
        vmin, vmax = float(values.min()), float(values.max())
    else:
        vmin, vmax = value_range
    scaled = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = (COLORMAP(scaled)[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return Image.fromarray(rgb)


def label_panel(image: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    label_h = 24
    canvas = Image.new("RGB", (image.width, image.height + label_h), color=(18, 18, 18))
    canvas.paste(image, (0, label_h))
    ImageDraw.Draw(canvas).text((8, 6), label, fill=(245, 245, 245), font=font)
    return canvas


def fit_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = int(round(image.height * width / image.width))
    return image.resize((width, height), Image.BILINEAR)


def save_preview(image_path: Path, prior: np.ndarray, pred: np.ndarray, gt_path: Path | None, path: Path) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    prior_valid = prior > 0
    pred_valid = np.isfinite(pred) & (pred > 0)

    depth_values = [pred[pred_valid]]
    gt = None
    if gt_path is not None and gt_path.exists():
        gt = load_depth(gt_path).squeeze().numpy()
        gt_valid = gt > 0
        depth_values.append(gt[gt_valid])
    shared = np.concatenate(depth_values)
    value_range = (float(shared.min()), float(shared.max()))

    panels = [
        label_panel(fit_width(Image.fromarray(image), 360), "RGB"),
        label_panel(fit_width(colorize_depth(prior, prior_valid, value_range), 360), "Sparse prior"),
        label_panel(fit_width(colorize_depth(pred, pred_valid, value_range), 360), "LDCM depth"),
    ]
    if gt is not None:
        panels.insert(1, label_panel(fit_width(colorize_depth(gt, gt > 0, value_range), 360), "GT depth"))

    canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), color=(18, 18, 18))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LDCM on a bundled sparse-depth demo sample.")
    parser.add_argument("--checkpoint", default="pkqbajng/LDCM", help="LDCM checkpoint path or Hugging Face repo id.")
    parser.add_argument("--moge", default="Ruicheng/moge-2-vits-normal", help="MoGe checkpoint path or Hugging Face repo id.")
    parser.add_argument("--sample", default="assets/sample_1", help="Sample directory with image.png and sparse_depth.npy.")
    parser.add_argument("--output-dir", default="outputs/demo", help="Directory for prediction outputs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = sample_dir / "image.png"
    sparse_path = sample_dir / "sparse_depth.npy"
    gt_path = sample_dir / "gt_depth.npy"

    device = torch.device(args.device)
    image = load_image(image_path).to(device)
    prior = load_depth(sparse_path).to(device)

    model = LDCMModel.from_pretrained(args.checkpoint, args.moge).to(device).eval()
    with torch.inference_mode():
        output = model.infer(image, prior)

    pred = output["depth_pred"][0, 0].detach().cpu().float().numpy()
    np.save(output_dir / "depth_pred.npy", pred)
    save_depth_uint16(pred, output_dir / "depth_pred_mm_uint16.png")
    save_preview(image_path, prior[0, 0].detach().cpu().numpy(), pred, gt_path, output_dir / "preview.png")

    print(f"Saved prediction to {output_dir}")


if __name__ == "__main__":
    main()
