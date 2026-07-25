# Segmentation model

`magic_touch.tflite` is Google's MagicTouch interactive-segmentation checkpoint.
It supplies a broad, prompt-guided arm proposal before tissue-boundary
refinement. The backend executes the TFLite graph directly with LiteRT on the
CPU, avoiding MediaPipe's GPU/GL initialization on headless hosted workers.

There is deliberately no geometric or guide-shaped mask fallback. If the model
cannot return a valid mask, the API labels the result as a fallback and analyzes
the full frame.

- Model source: https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/latest/magic_touch.tflite
- Documentation: https://developers.google.com/mediapipe/solutions/vision/interactive_segmenter
- License: Apache-2.0

The model is loaded only by the Python backend. It is not bundled into the
Android web container.
