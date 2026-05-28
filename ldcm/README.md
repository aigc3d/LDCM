# LDCM v1 Model Code

This folder contains the model code matching the released LDCM checkpoint.

The model entry is:

```python
from ldcm import LDCMModel

model = LDCMModel.from_pretrained(
    "pkqbajng/LDCM",
    moge_path="ckpts/moge-2-vits-normal/model.pt",
)
model.eval()

output = model.infer(image, prior)
```

`LDCMModel.from_pretrained` constructs the matching LDCM v1 architecture:

- LDCM backbone: DINOv2 ViT-B/14 with register tokens
- depth branch: CNN `DepthEncoder`
- decode head: V1 `Head`
- MoGe branch: loaded by MoGe's own `from_pretrained`

The loader accepts a Hugging Face repo id such as `pkqbajng/LDCM`, a local
checkpoint file, or a local directory containing `ldcm.pt`. Any `moge.*` tensors
in the checkpoint are skipped because MoGe is frozen and loaded independently.
The non-MoGe part is still checked strictly by default. The package root is the
model code itself:

```text
ldcm/
  ldcm.py
  depth_encoder.py
  head.py
  poisson_completion.py
  utils.py
  layers/
  moge/
```
