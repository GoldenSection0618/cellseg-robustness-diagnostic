# Environment Setup

This project uses one conda environment named `cell`.

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
  google-genai transformers
```

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

- SAM2 model checkpoints are not installed by these commands. Download the selected
  checkpoint later when implementing the SAM2 protocol.
- Gemini requires an API key at runtime, for example `GOOGLE_API_KEY`.
- For an 8 GB GPU, prefer small/base SAM2 variants, single-image inference,
  conservative YOLO batch sizes, and resizing or tiling for large images.
