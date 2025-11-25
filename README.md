# TorchModelLite

torchModelLite is a python tool for moving PyTorch models into TFLite format, yup, so you get your neural nets running straight on Android, iOS, or whatever odd IoT is in your lab. Big aim: Easy CPU compatibility, and yeah, there’s first steps for GPU and NPU if you want more speed later. We stick close to torch.export() and cover all the must-have Core ATen ops.

## How to Convert PyTorch Models
Here’s demo: take a resnet18 (pretuned on imagenet), swap it into TFLite for pocket devices.

```python
import torch
import torchvision
import torchModelLite

resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
sample_inputs = (torch.randn(1, 3, 224, 224),)
edge_model = torchModelLite.convert(resnet18.eval(), sample_inputs)
edge_model.export("resnet18.tflite")
```

- Colab notebook with steps for fast tryout (see /docs).
- Extra options? See advanced API details.

## Transformers for On-Device
Our Generative API: Write, quantize, and ship transformer models. Build LLMs for Android, et setera, then convert to TFLite. Tied up nice with MediaPipe LLM Inference API for direct app glue code. All docs under /docs, with practical walkthroughs.

*Note*: For now, CPU only, but GPU/NPU in pipeline. We sync with pytorch folk for future direct transformer support—no hacky rewrites.

## install
- Python >=3.10, Linux (for now)
- PyTorch: torch
- TensorFlow: tf-nightly

```bash
python -m venv --prompt torchModelLite venv
source venv/bin/activate
pip install torchModelLite
```
For bleeding-edge:
```bash
pip install torchModelLite-nightly
```
Release log and nightly drops at PyPi.

## contribute & questions
- Check CONTRIBUTING.md for PRs and code rules.
- Bugs, feedback, questions, log a github issue.
