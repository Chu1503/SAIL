# Segmentation model

`efficientsam_ti.onnx` is the EfficientSAM-Ti ONNX checkpoint used to obtain a
prompt-guided forearm core before boundary refinement.

- Project: https://github.com/yformer/EfficientSAM
- Model source: https://huggingface.co/yunyangx/EfficientSAM
- License: Apache-2.0
- Paper: EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything

The model is loaded only by the Python backend. It is not bundled into the
Android web container.
