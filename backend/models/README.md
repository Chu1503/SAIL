# Segmentation model

`magic_touch.tflite` is Google's MagicTouch interactive-segmentation checkpoint.
It supplies a broad, prompt-guided arm proposal before geometric and tissue
boundary refinement.

- Model source: https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/latest/magic_touch.tflite
- Documentation: https://developers.google.com/mediapipe/solutions/vision/interactive_segmenter
- License: Apache-2.0

The model is loaded only by the Python backend. It is not bundled into the
Android web container.
