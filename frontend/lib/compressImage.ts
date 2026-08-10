// Re-encodes a capture as JPEG, capped to a max dimension, before it goes
// over the network. The native USB camera path in particular hands back
// large, uncompressed frames straight from the sensor; the model pipeline
// resizes everything internally to well under 1600px anyway (SAM2 to 1024,
// CUBITAL to 512), so nothing analysis-relevant is lost, and the upload
// (and therefore round-trip latency) shrinks substantially.
const MAX_DIMENSION = 1600;
const JPEG_QUALITY = 0.85;

export async function compressImage(blob: Blob): Promise<Blob> {
  try {
    const bitmap = await createImageBitmap(blob);
    const scale = Math.min(1, MAX_DIMENSION / Math.max(bitmap.width, bitmap.height));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return blob;

    ctx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();

    const compressed = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY);
    });

    return compressed ?? blob;
  } catch (error) {
    console.error("Image compression failed, uploading original:", error);
    return blob;
  }
}
