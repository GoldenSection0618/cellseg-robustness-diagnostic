# Environment Setup

This project uses a primary conda environment named `cell`. The optional SAM3
direction uses a separate `sam3` environment so its Python and CUDA stack does
not change the established Cellpose, SAM2, and YOLO setup.

Create the environment:

```bash
conda create -n cell python=3.10 pip -y
```

Install the core scientific, image-processing, plotting, and evaluation stack:

```bash
mamba install -n cell -y -c conda-forge \
  numpy scipy pandas matplotlib-base seaborn-base \
  scikit-image scikit-learn imageio tifffile pycocotools \
  pyyaml tqdm requests psutil
```

Install PyTorch and torchvision:

```bash
conda run -n cell python -m pip install --retries 10 --timeout 120 \
  torch==2.9.1 torchvision==0.24.1
```

Install the segmentation, supervised training, and API packages:

```bash
conda run -n cell python -m pip install --retries 10 --timeout 120 \
  cellpose==4.1.1 sam2==1.1.0 ultralytics ultralytics-thop \
  pi-heif google-genai transformers
```

## Optional SAM3 Environment

Create the separate environment required by the current SAM3 release:

```bash
conda create -n sam3 python=3.12 pip -y
```

Install the GPU runtime and all declared or required inference dependencies
from conda-forge:

```bash
mamba install -n sam3 -y -c conda-forge \
  pytorch=2.12.1 pytorch-gpu=2.12.1 torchvision \
  numpy=1.26.4 timm tqdm ftfy=6.1.1 regex iopath typing_extensions \
  huggingface_hub pillow einops pycocotools psutil \
  imageio pandas scipy scikit-image
```

Install the upstream SAM3 package. This is the only required pip command,
because SAM3 is distributed as an upstream source repository rather than a
Conda package:

```bash
conda run -n sam3 python -m pip install --no-deps \
  git+https://github.com/facebookresearch/sam3.git@46957e47805eaa273f4aa7bbbd25a88bca9108ce
```

Verify the package and CUDA path:

```bash
conda run -n sam3 python -m pip check
conda run -n sam3 python -c "import sam3, torch; x=torch.ones((1024, 1024), device='cuda'); print('sam3', sam3.__version__); print('torch', torch.__version__, 'cuda', torch.version.cuda); print(torch.cuda.get_device_name(0), x.sum().item())"
```

SAM3 checkpoints require approved Hugging Face access and authentication; they
are model assets, not environment dependencies. This setup uses CUDA libraries
inside the conda environment and does not require or modify a system-level CUDA
toolkit.

## Verification

Check dependency consistency:

```bash
conda run -n cell python -m pip check
```

Expected output:

```text
No broken requirements found.
```

Check key imports and CUDA availability:

```bash
conda run -n cell python -c "import torch, cellpose, sam2, ultralytics, transformers; import google.genai; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('cellpose', cellpose.version); print('ultralytics', ultralytics.__version__); print('transformers', transformers.__version__)"
```

Run the Ultralytics environment check:

```bash
conda run -n cell yolo checks
```

Check that the microscopy image stack can read the DSB2018 data:

```bash
conda run -n cell python -c "from pathlib import Path; import imageio.v3 as iio; p=next(Path('data/raw/dsb2018/stage1_train').glob('*/images/*.png')); img=iio.imread(p); print(p); print(img.shape, img.dtype)"
```

## Notes

- SAM2 model checkpoints are local binary assets, not environment packages. The
  current SAM2 baseline expects `data/checkpoints/sam2.1_hiera_large.pt`.
- The current SAM2 install can run AMG inference, but its optional compiled `_C`
  extension is not importable, so SAM2 skips the optional post-processing step.
- Gemini requires an API key at runtime, for example `GOOGLE_API_KEY`.
- For an 8 GB GPU, prefer small/base SAM2 variants, single-image inference,
  conservative YOLO batch sizes, and resizing or tiling for large images.
